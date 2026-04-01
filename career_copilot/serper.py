"""Serper.dev wrapper — fetches company culture hints and synthesizes them with an LLM.

Expects SERPER_API_KEY in the environment.  The correct Serper endpoint is
https://google.serper.dev/search (overridable via SERPER_URL env var).
"""
import os
from typing import List, Any, Dict
import requests
from dotenv import load_dotenv

# Ensure .env is loaded even if this module is imported before config.py
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# Serper's actual endpoint — NOT serpapi.com which is a separate paid product
SERPER_URL = os.getenv("SERPER_URL", "https://google.serper.dev/search")


def _get_api_key() -> str:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise EnvironmentError("SERPER_API_KEY not set in environment")
    return key


def fetch_company_culture(company_name: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Query Serper.dev for company culture hints and return a list of snippets.

    Returns a list of dicts with keys: `title`, `snippet`, `link` when available.
    If the response format is unexpected, returns a single entry with the raw JSON
    under the `raw` key.
    """
    if not company_name:
        return []
    key = _get_api_key()
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    q = f"{company_name} company culture reviews employee experience Glassdoor"  # focused query
    payload = {"q": q, "num": num_results}

    resp = requests.post(SERPER_URL, json=payload, headers=headers, timeout=15)
    if resp.status_code == 403:
        raise EnvironmentError(
            "Serper API returned 403 Forbidden. "
            "Your SERPER_API_KEY may be invalid or the free quota is exhausted. "
            "Please check your account at https://serper.dev — the key should be a "
            "short alphanumeric string (not a hex hash)."
        )
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict[str, Any]] = []
    # Common Serper response shapes: 'organic' or 'results' may contain items
    candidates = []
    if isinstance(data, dict):
        if "organic" in data and isinstance(data["organic"], list):
            candidates = data["organic"]
        elif "results" in data and isinstance(data["results"], list):
            candidates = data["results"]
        elif "knowledge" in data and isinstance(data["knowledge"], list):
            candidates = data["knowledge"]

    if candidates:
        for item in candidates[:num_results]:
            title = item.get("title") or item.get("name")
            snippet = item.get("snippet") or item.get("text") or item.get("description")
            link = item.get("link") or item.get("url")
            results.append({"title": title, "snippet": snippet, "link": link})
        return results

    # Fallback: try to pull any top-level text fields or return raw
    flattened = []
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            flattened.append(obj)

    walk(data)
    if flattened:
        return [{"title": None, "snippet": s, "link": None} for s in flattened[:num_results]]

    return [{"raw": data}]


def synthesize_culture_insights(company_name: str, snippets: List[Dict[str, Any]]) -> str:
    """Pass raw search snippets to the LLM and return a structured culture insight.

    Returns a plain-text summary with four sections:
      - Core Values
      - Work Culture & Environment
      - What They Look For in Candidates
      - Suggested Talking Point (for cover letter / interview)
    """
    from career_copilot.config import get_client, get_model

    if not snippets:
        return f"No search results available for {company_name}."

    # Build a compact evidence block from the snippets
    evidence_lines = []
    for i, h in enumerate(snippets, 1):
        title = h.get("title") or ""
        snippet = h.get("snippet") or ""
        if title or snippet:
            evidence_lines.append(f"[{i}] {title}\n{snippet}".strip())
    evidence = "\n\n".join(evidence_lines[:6])

    prompt = (
        f"The following are web search snippets about {company_name}, "
        f"sourced from Glassdoor reviews, LinkedIn, and news articles.\n\n"
        f"{evidence}\n\n"
        "Based solely on the information above, write a concise company culture briefing "
        "for a job applicant preparing a cover letter and interview. "
        "Structure your response with exactly these four labelled sections:\n\n"
        "**Core Values:** (2-3 bullet points — what principles the company publicly emphasises)\n"
        "**Work Culture & Environment:** (2-3 bullet points — pace, teamwork style, expectations)\n"
        "**What They Look For in Candidates:** (2-3 bullet points — traits and skills they value)\n"
        "**Suggested Talking Point:** (1-2 sentences the applicant can adapt for their cover letter "
        "or use as a closing statement in an interview — must sound genuine, not generic)\n\n"
        "If the snippets lack enough information for a section, write \"Insufficient data\" for that section. "
        "Do not invent information not present in the snippets."
    )

    client = get_client()
    model = get_model("gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a career coach helping a student understand a target company."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()
