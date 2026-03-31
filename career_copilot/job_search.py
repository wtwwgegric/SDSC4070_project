"""Job Search — two independent utilities:

1. search_jobs(search_term, location, ...)
   Uses python-jobspy to search LinkedIn and/or Indeed and return a list of
   job posting dicts (title, company, location, description, job_url, …).

2. fetch_jd_from_url(url)
   Lightweight HTTP fetch for a single LinkedIn or Indeed job URL.
   Returns the description text, or raises ValueError with a user-friendly
   message if the page is blocked or can't be parsed.
"""
from __future__ import annotations

import re
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Shared browser-like headers to reduce bot detection
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ---------------------------------------------------------------------------
# 1.  JobSpy-based job search
# ---------------------------------------------------------------------------

def search_jobs(
    search_term: str,
    location: str = "",
    country_indeed: str = "",
    sites: list[str] | None = None,
    results_wanted: int = 10,
    hours_old: int = 168,           # last 7 days
    fetch_description: bool = True,
) -> list[dict[str, Any]]:
    """Search job boards and return a list of job posting dicts.

    Parameters
    ----------
    search_term      : Job title / keywords, e.g. "data analyst intern"
    location         : City / region, e.g. "Hong Kong". Leave empty for global.
    country_indeed   : Country name for Indeed/Glassdoor filtering (must match
                       JobSpy's valid country list, e.g. "hong kong", "usa").
                       If empty, defaults to location value.
    sites            : List from ["linkedin", "indeed"]. Default: both.
    results_wanted   : Max results *per site*. Capped at 10 for LinkedIn to
                       avoid rate-limit blocks.
    hours_old        : Only return jobs posted within this many hours.
    fetch_description: Fetch full description for LinkedIn (slower but
                       necessary for the JD Analyzer).

    Returns
    -------
    list of dicts with keys: title, company, location, job_type, date_posted,
                             job_url, description, site
    """
    from jobspy import scrape_jobs  # imported lazily to avoid startup cost

    if sites is None:
        sites = ["linkedin", "indeed"]

    # country_indeed falls back to location if not explicitly set
    _country = (country_indeed or location or "").strip().lower() or None

    # Cap LinkedIn at 10 to stay under its rate-limit threshold
    linkedin_cap = min(results_wanted, 10)
    indeed_cap = min(results_wanted, 25)

    rows = []
    for site in sites:
        cap = linkedin_cap if site == "linkedin" else indeed_cap
        kwargs: dict[str, Any] = dict(
            site_name=[site],
            search_term=search_term,
            location=location or None,
            results_wanted=cap,
            hours_old=hours_old,
            linkedin_fetch_description=(
                fetch_description and site == "linkedin"
            ),
            verbose=0,
        )
        if site == "indeed" and _country:
            kwargs["country_indeed"] = _country
        try:
            df = scrape_jobs(**kwargs)
            if df is not None and not df.empty:
                rows.append(df)
        except Exception:
            # If one site fails (rate-limited, invalid country, etc.) continue
            continue

    if not rows:
        return []

    import pandas as pd
    combined = pd.concat(rows, ignore_index=True)

    results: list[dict[str, Any]] = []
    for _, row in combined.iterrows():
        loc_parts = [
            str(row.get("city") or ""),
            str(row.get("state") or ""),
            str(row.get("country") or ""),
        ]
        loc_str = ", ".join(p for p in loc_parts if p and p != "nan")

        desc = str(row.get("description") or "")
        # Strip excessive whitespace left by markdown conversion
        desc = re.sub(r"\n{3,}", "\n\n", desc).strip()

        results.append({
            "title": str(row.get("title") or ""),
            "company": str(row.get("company") or ""),
            "location": loc_str,
            "job_type": str(row.get("job_type") or ""),
            "date_posted": str(row.get("date_posted") or ""),
            "job_url": str(row.get("job_url") or ""),
            "description": desc,
            "site": str(row.get("site") or site),
        })

    return results


# ---------------------------------------------------------------------------
# 2.  Direct URL fetch (LinkedIn + Indeed) — no JobSpy involved
# ---------------------------------------------------------------------------

_LINKEDIN_SELECTORS = [
    "div.description__text",
    "div.show-more-less-html__markup",
    "section.description div",
]

_INDEED_SELECTORS = [
    "div#jobDescriptionText",
    "div.jobsearch-jobDescriptionText",
]


def _parse_html(html: str, selectors: list[str]) -> str:
    """Extract text from the first matching CSS selector."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > 100:          # sanity: must have meaningful content
                return text
    return ""


def fetch_jd_from_url(url: str, timeout: int = 15) -> str:
    """Fetch the job description text from a LinkedIn or Indeed job URL.

    Returns the description as plain text.

    Raises
    ------
    ValueError  — with a user-friendly message if the page is blocked,
                  can't be parsed, or the URL is not supported.
    """
    url = url.strip()
    if not url.startswith("http"):
        raise ValueError("Please enter a full URL starting with https://")

    is_linkedin = "linkedin.com" in url
    is_indeed = "indeed.com" in url

    if not is_linkedin and not is_indeed:
        raise ValueError(
            "Only LinkedIn (linkedin.com/jobs) and Indeed (indeed.com) URLs are supported. "
            "For other sites please paste the JD text directly."
        )

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Network error fetching URL: {exc}") from exc

    if resp.status_code in (429, 999):
        raise ValueError(
            f"{'LinkedIn' if is_linkedin else 'Indeed'} blocked the request (HTTP {resp.status_code}). "
            "Please paste the JD text manually."
        )
    if resp.status_code == 404:
        raise ValueError("Job posting not found (404). It may have been removed.")
    if resp.status_code != 200:
        raise ValueError(
            f"Unexpected response (HTTP {resp.status_code}). Please paste the JD text manually."
        )

    selectors = _LINKEDIN_SELECTORS if is_linkedin else _INDEED_SELECTORS
    text = _parse_html(resp.text, selectors)

    if not text:
        raise ValueError(
            "Could not extract the job description from this page. "
            "LinkedIn/Indeed may have changed their page layout, or the page requires login. "
            "Please paste the JD text manually."
        )

    return text
