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


def hallucination_check(cv_text: str, generated_text: str, window: int = 4, jd_text: str = "") -> dict[str, Any]:
    """Multi-strategy grounding check for generated cover letters.

    Combines three complementary signals:
      1. Entity traceability — are proper nouns, numbers, and named entities in the
         generated text actually present in the CV?
      2. Short n-gram overlap — 3-to-4-word content phrases found verbatim in CV.
      3. Keyword recall — what fraction of distinctive CV keywords appear in the letter?

    The final grounding score is a weighted average (entity: 50%, n-gram: 25%, keyword: 25%)
    because entity traceability is the strongest hallucination signal (a fabricated company
    name or metric is the most damaging kind of hallucination in a cover letter).

    Returns:
      grounding_score       : float — weighted overall score (0-1, higher = better)
      entity_score          : float — fraction of generated entities found in CV
      ngram_score           : float — fraction of content n-grams found in CV
      keyword_score         : float — fraction of CV keywords used in letter
      total_entities        : int
      traceable_entities    : int
      untraceable_samples   : list[str] — entities/phrases not found in CV (for review)
      total_phrases         : int   — (legacy compat)
      traceable             : int   — (legacy compat)
      traceability_ratio    : float — (legacy compat, = grounding_score)
    """
    cv_lower = cv_text.lower()
    gen_lower = generated_text.lower()
    # JD text is a legitimate second source — entities from JD are not hallucinations
    jd_lower = jd_text.lower() if jd_text else ""

    # --- Strategy 1: Entity traceability ---
    # Extract things that MUST come from the CV: numbers, proper noun phrases,
    # company names, tool names, degree names, dates
    entity_patterns = [
        r'\b\d[\d,.%–-]+\b',                        # numbers, percentages, ranges
        r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Multi-word proper nouns
        r'\b(?:Python|SQL|C\+\+|Excel|Tableau|Pandas|NumPy|Scikit-learn|'
        r'Matplotlib|Seaborn|Jupyter|JSON|PCA|SVM|IQR|ML|KPI)\b',  # technical terms
    ]
    entities_found = set()
    for pat in entity_patterns:
        for m in re.finditer(pat, generated_text):
            ent = m.group().strip()
            if len(ent) >= 2:
                entities_found.add(ent)

    # Filter out common non-CV phrases, salutations, and generic professional language
    generic = {
        "Dear Hiring Manager", "Hiring Manager", "Dear Sir", "Dear Madam",
        "Yours Sincerely", "Yours Faithfully", "Kind Regards", "Best Regards",
        "Thank You", "Yours", "Thank", "Additionally", "Furthermore",
        "Complementing", "Technical Skills", "Work Experience", "Cover Letter",
        "Job Description", "Application Letter", "Career Objective",
        "Soft Skills", "Hard Skills", "Key Skills", "Core Competencies",
        "Bachelor", "Master", "Doctor", "Honours", "First Class",
    }
    # Also filter single-word entries and anything that looks like a common English phrase
    entities_found = {
        e for e in entities_found
        if e not in generic and len(e) > 2
        and not any(g.lower() in e.lower() for g in {"Hiring Manager", "Dear", "Sincerely", "Regards"})
    }

    traceable_entities = []
    untraceable_entities = []   # not in CV or JD → genuine hallucination risk
    jd_sourced_entities = []    # not in CV but in JD → legitimate inclusions
    for ent in entities_found:
        ent_lower = ent.lower()
        in_cv = ent_lower in cv_lower
        in_jd = bool(jd_lower) and ent_lower in jd_lower
        if in_cv:
            traceable_entities.append(ent)
        elif in_jd:
            jd_sourced_entities.append(ent)
        else:
            untraceable_entities.append(ent)

    # entity_score: only truly untraceable entities (not in CV AND not in JD) count as issues
    # Denominator excludes JD-sourced entities (they are legitimate, not hallucinations)
    cv_or_jd_checkable = len(entities_found) - len(jd_sourced_entities)
    if cv_or_jd_checkable > 0:
        entity_score = len(traceable_entities) / cv_or_jd_checkable
    else:
        entity_score = 1.0

    # --- Strategy 2: Short content n-gram overlap ---
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "as", "is", "am", "are", "was", "were", "i", "my", "your",
                 "this", "that", "these", "those", "by", "from", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
                 "shall", "should", "may", "might", "not", "no", "also", "very", "more"}

    words = gen_lower.split()
    ngram_traceable = 0
    ngram_total = 0
    for w in (3, 4):
        if len(words) < w:
            continue
        for i in range(len(words) - w + 1):
            gram = words[i: i + w]
            content_words = [t for t in gram if t.strip(string.punctuation) not in stopwords]
            if len(content_words) < 2:
                continue
            ngram_total += 1
            phrase = " ".join(gram)
            if phrase in cv_lower:
                ngram_traceable += 1

    ngram_score = (ngram_traceable / ngram_total) if ngram_total else 1.0

    # --- Strategy 3: Keyword recall (distinctive CV words used in letter) ---
    cv_words = set(re.findall(r'\b[a-z]{4,}\b', cv_lower))
    # Remove very common English words
    common = {"this", "that", "with", "from", "have", "been", "were", "also", "will",
              "more", "your", "they", "their", "about", "would", "could", "should",
              "which", "these", "those", "than", "into", "most", "such", "some",
              "each", "make", "like", "just", "over", "only", "very", "when", "what",
              "them", "then", "here", "where", "much", "well", "back", "after", "before",
              "through", "between", "under"}
    cv_distinctive = cv_words - common
    if cv_distinctive:
        gen_words = set(re.findall(r'\b[a-z]{4,}\b', gen_lower))
        keyword_overlap = len(cv_distinctive & gen_words)
        keyword_score = keyword_overlap / len(cv_distinctive)
    else:
        keyword_score = 1.0

    # --- Weighted final score ---
    grounding_score = round(
        0.50 * entity_score + 0.25 * ngram_score + 0.25 * keyword_score, 4
    )

    # Legacy-compatible keys
    return {
        "grounding_score": grounding_score,
        "entity_score": round(entity_score, 4),
        "ngram_score": round(ngram_score, 4),
        "keyword_score": round(keyword_score, 4),
        # Clear hallucination summary
        "total_entities": len(entities_found),
        "traceable_entities": len(traceable_entities),
        "jd_sourced_entities": len(jd_sourced_entities),   # legitimate JD inclusions
        "hallucinated_count": len(untraceable_entities),    # truly suspect: not in CV or JD
        "untraceable_samples": untraceable_entities[:5],
        # Legacy keys for app.py compatibility
        "total_phrases": len(entities_found),
        "traceable": len(traceable_entities),
        "traceability_ratio": grounding_score,
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
