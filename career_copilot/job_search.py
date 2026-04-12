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

from career_copilot.config import _get


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


def _search_jobs_via_serper(
    search_term: str,
    location: str = "",
    site: str = "linkedin",
    num_results: int = 10,
) -> list[dict[str, Any]]:
    """Fallback search using Serper web results.

    This is primarily used for Streamlit Cloud, where direct scraping from
    LinkedIn/Indeed is more likely to be blocked.
    """
    api_key = _get("SERPER_API_KEY")
    if not api_key:
        return []

    domain = "linkedin.com/jobs/view" if site == "linkedin" else "indeed.com"
    location_clause = f' "{location}"' if location.strip() else ""
    query = f'site:{domain} "{search_term}"{location_clause}'

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": min(num_results, 10)},
        timeout=15,
    )
    if resp.status_code != 200:
        return []

    data = resp.json()
    organic = data.get("organic", []) if isinstance(data, dict) else []
    results: list[dict[str, Any]] = []

    for item in organic[:num_results]:
        link = item.get("link") or ""
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        if not link:
            continue

        company = ""
        if " - " in title:
            parts = [p.strip() for p in title.split(" - ") if p.strip()]
            if len(parts) >= 2:
                title = parts[0]
                company = parts[1]

        results.append({
            "title": title,
            "company": company,
            "location": location,
            "job_type": "",
            "date_posted": "",
            "job_url": link,
            "description": snippet,
            "site": site,
        })

    return results


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

    # When no explicit city/location is given, use country/region as the
    # location hint so LinkedIn still gets a geographic filter.
    _effective_location = location or country_indeed or ""

    # Cap LinkedIn at 10 to stay under its rate-limit threshold
    linkedin_cap = min(results_wanted, 10)
    indeed_cap = min(results_wanted, 25)

    rows = []
    site_errors: list[str] = []
    fallback_results: list[dict[str, Any]] = []
    for site in sites:
        cap = linkedin_cap if site == "linkedin" else indeed_cap
        kwargs: dict[str, Any] = dict(
            site_name=[site],
            search_term=search_term,
            location=_effective_location or None,
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
                continue

            # LinkedIn sometimes returns empty on the first attempt due to
            # transient rate-limiting. Retry once without location filter.
            if site == "linkedin":
                import time
                time.sleep(2)
                retry_kw = dict(kwargs)
                retry_kw["location"] = None
                retry_df = scrape_jobs(**retry_kw)
                if retry_df is not None and not retry_df.empty:
                    rows.append(retry_df)
                    continue

            # Indeed often returns no rows on cloud runners for narrow geo filters.
            # Retry once with broader parameters before giving up.
            if site == "indeed":
                retry_kwargs = dict(kwargs)
                retry_kwargs["location"] = None
                retry_kwargs.pop("country_indeed", None)
                retry_df = scrape_jobs(**retry_kwargs)
                if retry_df is not None and not retry_df.empty:
                    rows.append(retry_df)
                    continue

            # Cloud-friendly fallback: use Serper web search to return job links.
            fallback_results.extend(
                _search_jobs_via_serper(
                    search_term=search_term,
                    location=_effective_location,
                    site=site,
                    num_results=cap,
                )
            )
        except Exception:
            # If one site fails (rate-limited, invalid country, etc.) continue
            site_errors.append(site)
            fallback_results.extend(
                _search_jobs_via_serper(
                    search_term=search_term,
                    location=_effective_location,
                    site=site,
                    num_results=cap,
                )
            )
            continue

    if not rows:
        if fallback_results:
            seen_urls: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for item in fallback_results:
                url = item.get("job_url") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                deduped.append(item)
            if deduped:
                return deduped
        if site_errors and len(site_errors) == len(sites):
            raise ValueError(
                "Job board search failed on all selected sites. "
                "LinkedIn may be rate-limited — wait a minute and retry, or reduce Results per site. "
                "If you added a Serper API key, the app will try a web-search fallback automatically. "
                "You can also paste a JD URL or text directly in the JD Analyzer tab."
            )
        raise ValueError(
            "No jobs were returned by the selected job boards. "
            "Try broader keywords, change the Country/Region filter, "
            "or paste a JD URL/text in the JD Analyzer tab."
        )

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


def _parse_structured_job_data(html: str) -> str:
    """Extract JD text from common JSON-LD / structured data blocks."""
    import json

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            desc = item.get("description")
            if isinstance(desc, str):
                text = re.sub(r"<[^>]+>", " ", desc)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"\s{2,}", " ", text).strip()
                if len(text) > 100:
                    return text
    return ""


def _fetch_via_reader(url: str, timeout: int = 30) -> str:
    """Fallback page reader for cloud environments where direct scraping is blocked."""
    reader_url = f"https://r.jina.ai/http://{url}"
    try:
        resp = requests.get(reader_url, timeout=timeout)
    except requests.exceptions.RequestException:
        return ""
    if resp.status_code != 200:
        return ""

    text = resp.text.strip()
    if "Markdown Content:" in text:
        text = text.split("Markdown Content:", 1)[1].strip()

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("Title:") or stripped.startswith("URL Source:"):
            continue
        if stripped.startswith("[Skip to main content]"):
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned if len(cleaned) > 100 else ""


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

    if is_linkedin and "/jobs/search/" in url:
        raise ValueError(
            "Please use a direct LinkedIn job page URL (usually contains /jobs/view/), not a search results page."
        )
    if is_indeed and re.search(r"/jobs(?:\.html)?(?:\?|$)", url):
        raise ValueError(
            "Please use a direct Indeed job posting URL, not an Indeed search results page."
        )

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        text = _fetch_via_reader(url)
        if text:
            return text
        raise ValueError(f"Network error fetching URL: {exc}") from exc

    if resp.status_code in (429, 999):
        text = _fetch_via_reader(url)
        if text:
            return text
        raise ValueError(
            f"{'LinkedIn' if is_linkedin else 'Indeed'} blocked the request (HTTP {resp.status_code}). "
            "Try the direct job URL, or paste the JD text manually."
        )
    if resp.status_code == 404:
        raise ValueError("Job posting not found (404). It may have been removed.")
    if resp.status_code != 200:
        text = _fetch_via_reader(url)
        if text:
            return text
        raise ValueError(
            f"Unexpected response (HTTP {resp.status_code}). Please paste the JD text manually."
        )

    selectors = _LINKEDIN_SELECTORS if is_linkedin else _INDEED_SELECTORS
    text = _parse_html(resp.text, selectors)
    if not text:
        text = _parse_structured_job_data(resp.text)
    if not text:
        text = _fetch_via_reader(url)

    if not text:
        raise ValueError(
            "Could not extract the job description from this page. "
            "LinkedIn/Indeed may have changed their page layout, the page may require login, "
            "or the URL may be a search page instead of a direct job post. "
            "Please paste the JD text manually."
        )

    return text
