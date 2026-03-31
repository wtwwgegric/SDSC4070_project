"""Simple Serper.dev wrapper to fetch company culture hints.

This module expects a Serper API key in the environment variable `SERPER_API_KEY`.
Default endpoint is `https://serpapi.com/search?engine=google` but can be overridden with
`SERPER_URL` env var.
"""
import os
from typing import List, Any, Dict
import requests
from dotenv import load_dotenv

# Ensure .env is loaded even if this module is imported before config.py
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

SERPER_URL = os.getenv("SERPER_URL", "https://serpapi.com/search?engine=google")


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
