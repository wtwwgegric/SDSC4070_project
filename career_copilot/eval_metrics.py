"""Evaluation Metrics for the Career Co-pilot project report.

Provides objective, quantifiable measures to include in the project report.

Metrics
-------
keyword_hit_rate(jd_text, cv_text) -> float
    Fraction of JD hard keywords that appear in the CV text (case-insensitive).

keyword_hit_rate_improvement(jd_text, cv_before, cv_after) -> dict
    Computes hit rate before and after CV optimisation and the absolute gain.

hallucination_check(cv_text, cover_letter_text) -> dict
    Checks which factual phrases in the cover letter are traceable to the CV.
    Returns counts and a traceability ratio (higher = less hallucination).

rubric_summary(scores: list[RubricScore]) -> dict
    Aggregates interview rubric scores into a per-round table suitable for
    plotting in Streamlit (st.line_chart).
"""
from __future__ import annotations

import re
import string
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def _extract_keywords_from_jd(jd_analysis: dict[str, Any]) -> list[str]:
    """Pull hard skills from jd_analysis dict."""
    return [k.lower().strip() for k in jd_analysis.get("hard_skills", [])]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def keyword_hit_rate(jd_analysis: dict[str, Any], cv_text: str) -> float:
    """Return fraction of JD hard-skill keywords found (case-insensitive) in cv_text.

    Returns 0.0 if there are no keywords to check.
    """
    keywords = _extract_keywords_from_jd(jd_analysis)
    if not keywords:
        return 0.0
    cv_lower = cv_text.lower()
    hits = sum(1 for kw in keywords if kw in cv_lower)
    return round(hits / len(keywords), 4)


def keyword_hit_rate_improvement(
    jd_analysis: dict[str, Any],
    cv_before: str,
    cv_after: str,
) -> dict[str, Any]:
    """Compare keyword hit rates before and after CV optimisation.

    Returns a dict with:
      keywords        : list[str] — the JD keywords checked
      rate_before     : float
      rate_after      : float
      absolute_gain   : float
      relative_gain   : float — (after - before) / before, or None if before == 0
      per_keyword     : list[dict] — per-keyword found_before / found_after flags
    """
    keywords = _extract_keywords_from_jd(jd_analysis)
    before_lower = cv_before.lower()
    after_lower = cv_after.lower()

    per_keyword = []
    hits_before = 0
    hits_after = 0
    for kw in keywords:
        fb = kw in before_lower
        fa = kw in after_lower
        hits_before += int(fb)
        hits_after += int(fa)
        per_keyword.append({"keyword": kw, "found_before": fb, "found_after": fa})

    n = len(keywords) or 1
    rate_before = round(hits_before / n, 4)
    rate_after = round(hits_after / n, 4)
    abs_gain = round(rate_after - rate_before, 4)
    rel_gain = round((rate_after - rate_before) / rate_before, 4) if rate_before else None

    return {
        "keywords": keywords,
        "rate_before": rate_before,
        "rate_after": rate_after,
        "absolute_gain": abs_gain,
        "relative_gain": rel_gain,
        "per_keyword": per_keyword,
    }


def hallucination_check(cv_text: str, generated_text: str, window: int = 6) -> dict[str, Any]:
    """Heuristic hallucination check: are n-gram phrases in the generated text traceable to the CV?

    Method: extract all `window`-word n-grams from the generated text, check if each
    appears verbatim in the CV. This is a conservative lower bound but is fully objective.

    Returns:
      total_phrases    : int
      traceable        : int
      traceability_ratio: float — higher is better (less hallucination)
      untraceable_samples: list[str] — up to 5 unverified phrases for manual review
    """
    cv_lower = cv_text.lower()
    gen_lower = generated_text.lower()

    words = gen_lower.split()
    if len(words) < window:
        return {"total_phrases": 0, "traceable": 0, "traceability_ratio": 1.0, "untraceable_samples": []}

    ngrams = [" ".join(words[i: i + window]) for i in range(len(words) - window + 1)]
    # Only check noun-like phrases (skip those with mostly stopwords)
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "as", "is", "are", "was", "were", "i", "my", "your"}

    def _is_content_ngram(ng: str) -> bool:
        tokens = ng.split()
        content = [t for t in tokens if t not in stopwords]
        return len(content) >= 2

    content_ngrams = [ng for ng in ngrams if _is_content_ngram(ng)]
    if not content_ngrams:
        return {"total_phrases": 0, "traceable": 0, "traceability_ratio": 1.0, "untraceable_samples": []}

    traceable = [ng for ng in content_ngrams if ng in cv_lower]
    untraceable = [ng for ng in content_ngrams if ng not in cv_lower]

    ratio = round(len(traceable) / len(content_ngrams), 4) if content_ngrams else 1.0

    return {
        "total_phrases": len(content_ngrams),
        "traceable": len(traceable),
        "traceability_ratio": ratio,
        "untraceable_samples": untraceable[:5],
    }


def rubric_summary(scores: list) -> list[dict[str, Any]]:
    """Convert a list of RubricScore objects into a list of dicts for Streamlit charting.

    Compatible with st.line_chart (index = round number).
    """
    return [
        {
            "Round": i + 1,
            "Technical Accuracy": s.technical_accuracy,
            "Completeness": s.completeness,
            "Clarity": s.clarity,
            "Overall": s.overall,
        }
        for i, s in enumerate(scores)
    ]
