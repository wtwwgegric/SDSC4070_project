"""Cover Letter Generator — produces a targeted cover letter from JD analysis + CV matches.

Rules enforced in the prompt:
- Every claim must be traceable to the full CV text provided.
- No fabricated numbers, job titles, or achievements.
- Output is a ready-to-send cover letter in plain English.
"""
import re
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
    "[Candidate School], I am eager to apply my analytical skills and passion for "
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
    "STRICT RULE: Every claim, achievement, or metric you mention MUST be present in the CV "
    "text provided. Never invent facts not in the source material."
)

# Section header keywords used to label CV sections
_SECTION_PATTERNS = [
    (r"(work\s*experience|professional\s*experience|employment)", "WORK EXPERIENCE"),
    (r"(internship|intern\s*experience)", "INTERNSHIP"),
    (r"(project|projects|personal\s*project|academic\s*project)", "PROJECTS"),
    (r"(education|academic\s*background|qualification)", "EDUCATION"),
    (r"(skill|skills|technical\s*skill|core\s*competenc)", "SKILLS"),
    (r"(award|achievement|honour|honor|certification)", "AWARDS & CERTIFICATIONS"),
    (r"(volunteer|extracurricular|activity|activities|leadership)", "ACTIVITIES"),
    (r"(summary|profile|objective|about\s*me)", "PROFILE SUMMARY"),
]


def _label_cv_sections(cv_text: str) -> str:
    """Split CV text on blank lines, tag lines that look like section headers,
    and return an annotated version the LLM can use to understand structure."""
    paragraphs = re.split(r"\n\s*\n", cv_text)
    annotated: list[str] = []
    current_section = "GENERAL"

    for para in paragraphs:
        first_line = para.strip().split("\n")[0].strip()
        # Short ALL-CAPS or title-cased short line → likely a section header
        is_header = (
            len(first_line) < 60
            and (first_line.isupper() or re.match(r"^[A-Z][a-zA-Z &/]+$", first_line))
        )
        if is_header:
            # Only update section when the header matches a known pattern.
            # Unrecognised short lines (role titles, company names) keep the
            # current section so they stay tagged correctly.
            for pattern, label in _SECTION_PATTERNS:
                if re.search(pattern, first_line, re.IGNORECASE):
                    current_section = label
                    break

        annotated.append(f"[{current_section}]\n{para.strip()}")

    return "\n\n".join(annotated)


def generate_cover_letter(
    jd_analysis: dict[str, Any],
    match_results: list[dict[str, Any]],
    candidate_name: str = "the candidate",
    company_name: str = "your company",
    cv_text: str = "",
    model: str = None,
) -> str:
    """Generate a cover letter grounded in the full CV text.

    Parameters
    ----------
    jd_analysis   : output of `analyze_jd()`
    match_results : output of `match_cv_to_jd()` — used to identify most relevant keywords
    candidate_name: name to use in the opening line
    company_name  : target company for the letter
    cv_text       : full raw CV text (strongly recommended — enables section-aware writing)
    model         : model override

    Returns
    -------
    str — full cover letter text
    """
    if not match_results and not cv_text:
        raise ValueError("Provide cv_text or match_results — both are empty")

    client = get_client()
    model = model or get_model("gpt-4o-mini")

    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:8])
    soft_skills = ", ".join(jd_analysis.get("soft_skills", [])[:5])
    role_summary = jd_analysis.get("summary", "the role")
    interview_topics = "; ".join(jd_analysis.get("interview_topics", [])[:5])

    # Build the CV context block — prefer full labelled CV over short snippets
    if cv_text:
        labelled_cv = _label_cv_sections(cv_text)
        cv_block = (
            "FULL CV (sections labelled — use this as your primary source):\n"
            "---\n"
            f"{labelled_cv[:4000]}\n"
            "---"
        )
    else:
        # Fallback: use RAG snippets only
        lines = [f"[{i}] {m.get('suggested_rewrite') or m.get('original', '')}"
                 for i, m in enumerate(match_results[:8], 1)]
        cv_block = "CV EVIDENCE SNIPPETS (RAG-retrieved):\n" + "\n".join(lines)

    prompt = (
        f"Write a professional cover letter for {candidate_name} applying to {company_name}.\n\n"
        f"ROLE SUMMARY: {role_summary}\n"
        f"KEY HARD SKILLS REQUIRED: {hard_skills}\n"
        f"KEY SOFT SKILLS REQUIRED: {soft_skills}\n"
        f"LIKELY INTERVIEW TOPICS: {interview_topics}\n\n"
        f"{cv_block}\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the CV sections carefully. Distinguish between:\n"
        "   - [WORK EXPERIENCE] / [INTERNSHIP]: real-world professional roles — treat these as "
        "the candidate's strongest credibility evidence.\n"
        "   - [PROJECTS]: self-initiated or academic projects — use to demonstrate technical skills.\n"
        "   - [EDUCATION]: degree, institution, GPA/awards if mentioned.\n"
        "2. Write the letter in this structure:\n"
        "   a. 'Dear Hiring Manager,' then 1–2 sentences: role applied for + candidate's "
        "current status (degree, year, institution from CV) + one-line motivation.\n"
        "   b. Body paragraph 1: Draw from [INTERNSHIP] or [WORK EXPERIENCE] — describe one "
        "specific role, what the candidate did, and how it maps to the JD.\n"
        "   c. Body paragraph 2: Draw from [PROJECTS] or a second distinct experience — "
        "highlight a technical skill or achievement relevant to the JD hard skills.\n"
        "   d. Closing: 1–2 sentences of genuine enthusiasm for the company/role. "
        "Polite call-to-action.\n"
        "   e. 'Yours faithfully,' then the candidate's name.\n"
        "3. Use specific details (role titles, company names, tools, outcomes) exactly as they "
        "appear in the CV. Do NOT generalise or invent.\n"
        "4. Target length: 350–420 words. Full sentences, no bullet points."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()

