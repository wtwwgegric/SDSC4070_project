"""Cover Letter Generator — produces a targeted cover letter from JD analysis + CV matches.

Rules enforced in the prompt:
- Every claim must be traceable to the CV passages provided.
- No fabricated numbers, job titles, or achievements.
- Output is a ready-to-send cover letter in plain English.
"""
import os
from typing import Any

from career_copilot.config import get_client, get_model


_SYSTEM = (
    "You are an expert cover letter writer. "
    "Write in a professional yet warm first-person voice. "
    "STRICT RULE: Every claim, achievement, or metric you mention MUST come from "
    "the CV excerpts provided. Never invent facts not present in the source material."
)


def _format_cv_evidence(match_results: list[dict[str, Any]]) -> str:
    lines = []
    for i, m in enumerate(match_results, 1):
        lines.append(f"[{i}] {m.get('suggested_rewrite') or m.get('original', '')}")
    return "\n".join(lines)


def generate_cover_letter(
    jd_analysis: dict[str, Any],
    match_results: list[dict[str, Any]],
    candidate_name: str = "the candidate",
    company_name: str = "your company",
    model: str = None,
) -> str:
    """Generate a cover letter grounded in CV evidence.

    Parameters
    ----------
    jd_analysis   : output of `analyze_jd()`
    match_results : output of `match_cv_to_jd()`
    candidate_name: name to use in the opening line
    company_name  : target company for the letter
    model         : OpenAI model override

    Returns
    -------
    str — full cover letter text
    """
    if not match_results:
        raise ValueError("match_results is empty — run match_cv_to_jd() first")

    client = _client()
    model = model or os.getenv("OPENAI_MODEL", "qwen3.5-plus")

    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:8])
    role_summary = jd_analysis.get("summary", "the role")
    cv_evidence = _format_cv_evidence(match_results[:6])

    prompt = (
        f"Write a cover letter for {candidate_name} applying to {company_name}.\n\n"
        f"Role summary: {role_summary}\n\n"
        f"Key skills the JD emphasises: {hard_skills}\n\n"
        f"CV evidence to draw from (use these; do not add anything extra):\n{cv_evidence}\n\n"
        "Structure: opening hook (1 sentence) → 2 body paragraphs mapping skills to JD → "
        "closing call to action. Keep it under 350 words."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()
