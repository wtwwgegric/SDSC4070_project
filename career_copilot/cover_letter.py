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
    "Write in a professional yet warm first-person voice, modelled on the style shown "
    "in the examples below.\n\n"
    "STYLE EXAMPLES (do not copy verbatim — use as tone and structure reference):\n\n"
    "--- EXAMPLE 1 ---\n"
    "Dear Hiring Manager,\n"
    "I am writing to express my strong interest in the 2026 Cathay Academy Summer Intern "
    "position focusing on Digital Learning and Analytics. As a Data Science undergraduate at "
    "Chinese University of Hong Kong, I am eager to apply my analytical skills and passion for "
    "technology to support innovative learning solutions within Cathay Academy.\n"
    "My academic training and project experiences have equipped me with a solid foundation in "
    "data analytics, automation, and visualization. During my internship at China Mobile Hong Kong, "
    "I structured knowledge for an AI chatbot by converting operational materials into standardized "
    "QA triplets, and evaluated chatbot intent classification accuracy by testing responses against "
    "natural language variations. This experience strengthened my ability to work with AI applications "
    "and identify areas for improvement through data-driven insights.\n"
    "Additionally, I have developed strong skills in data processing and visualization. Through my "
    "financial market trend analysis project, I used Python to analyze historical data and created "
    "interactive dashboards using Tableau. These experiences align closely with the responsibilities "
    "of streamlining analytical processes, building automated dashboards, and supporting data analytics "
    "within the Learning Management System revamp.\n"
    "I am drawn to Cathay Academy's commitment to leveraging technology for continuous learning and "
    "innovation. I am a proactive learner, collaborative by nature, and passionate about applying data "
    "analytics to solve real-world problems.\n"
    "Thank you for considering my application. I have attached my CV for your review and would be "
    "delighted to discuss how I can contribute to your team.\n"
    "Yours faithfully,\n"
    "[Candidate Name]\n\n"
    "--- EXAMPLE 2 ---\n"
    "Dear Recruitment Manager,\n"
    "I am writing to apply for the administrative and operational support role at China Mobile Hong Kong. "
    "As a Data Science undergraduate, I am eager to contribute my analytical skills, bilingual proficiency, "
    "and attention to detail to support the company's mission. My academic training and hands-on experience "
    "in data analysis, report generation, and cross-functional collaboration align closely with the "
    "responsibilities outlined in this role.\n"
    "During my studies, I have developed a strong foundation in data processing and documentation. In my "
    "Visualization Project, I analyzed large datasets using Excel and Python to identify trends and created "
    "interactive Tableau dashboards to communicate findings. This highlights my ability to transform raw "
    "data into actionable insights.\n"
    "My technical proficiency extends to Python-based solutions. In a Fraud Detection project, I designed "
    "a machine learning pipeline to classify transactional anomalies, emphasizing meticulous data cleaning "
    "and validation. This underscores my capacity to manage detail-oriented tasks independently.\n"
    "I am confident my analytical mindset, adaptability, and dedication to precision would enable me to "
    "contribute meaningfully to your team's success.\n"
    "Thank you for considering my application.\n"
    "Yours faithfully,\n"
    "[Candidate Name]\n"
    "--- END EXAMPLES ---\n\n"
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

    client = get_client()
    model = model or get_model("gpt-4o-mini")

    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:8])
    role_summary = jd_analysis.get("summary", "the role")
    cv_evidence = _format_cv_evidence(match_results[:6])

    prompt = (
        f"Write a cover letter for {candidate_name} applying to {company_name}.\n\n"
        f"Role summary: {role_summary}\n\n"
        f"Key skills the JD emphasises: {hard_skills}\n\n"
        f"CV evidence to draw from (use these; do not add anything extra):\n{cv_evidence}\n\n"
        "Required structure (follow the style of the examples in your system prompt):\n"
        "1. Opening: 'Dear Hiring Manager,' salutation, then 1-2 sentences stating the role "
        "applied for and the candidate's current academic/professional status plus motivation.\n"
        "2. Body paragraph 1: 1 specific past experience or project from the CV evidence that "
        "directly maps to a key JD requirement. Include what the candidate did and the outcome.\n"
        "3. Body paragraph 2: A second distinct experience or skill set from the CV evidence "
        "that addresses another JD requirement.\n"
        "4. Closing paragraph: 1-2 sentences expressing enthusiasm for the company/role "
        "and a polite call-to-action.\n"
        "5. Sign off: 'Yours faithfully,' then the candidate's name on a new line.\n\n"
        "Keep it under 400 words. Use full sentences, no bullet points."
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
