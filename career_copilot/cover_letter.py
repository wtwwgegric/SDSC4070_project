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
    "You are a professional career consultant who writes tailored cover letters "
    "for students and early-career candidates. You work for any discipline "
    "(technical, business, creative, hybrid). You infer the candidate's profile "
    "from the CV provided and align it to the JD.\n\n"

    "=== INTERNAL REASONING PIPELINE (do NOT print these steps) ===\n"
    "Before writing, silently execute these steps in order:\n"
    "  Step 1   CV Parsing: identify the candidate's degree, institution, year, "
    "work/internship roles, projects, skills, and languages.\n"
    "  Step 2   JD Understanding: extract the 3-5 core responsibilities and "
    "the hard/soft skills the employer values most.\n"
    "  Step 3   Relevance Scoring: mentally rank every CV experience by how "
    "directly it demonstrates the JD's core requirements.\n"
    "  Step 4   Experience Selection: pick the Top-2 or Top-3 most relevant "
    "experiences. Prefer [WORK EXPERIENCE] / [INTERNSHIP] over [PROJECTS] "
    "when both exist. Never pick more than 3.\n"
    "  Step 5   Semantic Mapping: for each selected experience, identify the "
    "specific actions, tools, and outcomes that map to a JD requirement. "
    "Rewrite them in the candidate's own voice, do NOT copy JD phrases.\n"
    "  Step 6   Controlled Generation: produce the letter following the "
    "structure and style rules below.\n\n"

    "=== OUTPUT STRUCTURE (5 paragraphs, strict) ===\n"
    "1. OPENING (2-3 sentences): state the role and company name, who the "
    "candidate is (school + major or current role), and one genuine motivation.\n"
    "2. BODY 1 (3-5 sentences): the single most relevant work/internship "
    "experience. Describe what the candidate DID, HOW, and what OUTCOME it "
    "produced. Map it explicitly to one JD requirement.\n"
    "3. BODY 2 (3-5 sentences): a second distinct experience (project, "
    "another role, or complementary skill). Map it to a different JD "
    "requirement. If the candidate has only academic projects, that is fine.\n"
    "4. INTEREST & FIT (2-3 sentences): express genuine enthusiasm for the "
    "company or industry. Mention a soft skill with brief evidence. If the "
    "JD asks about availability or language, state it here.\n"
    "5. CLOSING (2-3 sentences): thank the reader, mention the attached CV, "
    "express willingness to discuss further. Sign off with 'Yours faithfully,' "
    "then the candidate's name.\n\n"

    "=== WRITING STYLE CONSTRAINTS (MUST obey) ===\n"
    "- NO dashes: never use '--' or '\u2014'.\n"
    "- NO AI-style parallelism: avoid 'not only... but also...', "
    "'both... and...', 'as well as' chains.\n"
    "- VARY sentence length: mix short punchy sentences with longer ones.\n"
    "- NO adjective piling: never write 'passionate, diligent, innovative' "
    "without evidence. Replace with verbs and facts.\n"
    "- DO NOT copy the CV verbatim: reinterpret experiences in natural prose.\n"
    "- DO NOT echo JD wording: rephrase requirements in the candidate's voice.\n"
    "- Tone: confident but not arrogant. 'I am confident I can contribute' is "
    "good. 'I am the perfect fit' is bad.\n"
    "- Target 280-400 words. Full sentences, no bullet points.\n\n"

    "=== CONTENT SELECTION RULES ===\n"
    "INCLUDE: experiences directly relevant to JD; concrete actions and outcomes; "
    "language abilities if JD mentions them; availability if JD asks.\n"
    "EXCLUDE: courses unless JD explicitly requires them; basic skills the JD "
    "does not mention (e.g. 'Microsoft Word' for a software engineer role); "
    "experiences that have zero connection to the role.\n"
    "EDUCATION: degree + school + major + year only. GPA only if strong.\n\n"

    "=== SELF-CHECK (verify before returning) ===\n"
    "[ ] Company name and role title appear in the opening paragraph.\n"
    "[ ] At least 2 specific CV experiences are used with concrete details.\n"
    "[ ] No dashes, no parallel phrasing, varied sentence lengths.\n"
    "[ ] No fabricated facts, numbers, or company names.\n"
    "[ ] Closing includes thanks + attached CV + willingness to discuss.\n"
    "[ ] Signed 'Yours faithfully,' then the candidate's name.\n\n"

    "=== GROUNDING RULE ===\n"
    "Every factual claim (company name, job title, tool, metric, outcome) MUST "
    "appear in the CV text provided. You may infer soft skills from described "
    "behaviours, but never invent hard facts."
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
        f"Generate a cover letter for {candidate_name} applying to {company_name}.\n\n"
        f"=== JOB DESCRIPTION ANALYSIS ===\n"
        f"Role summary: {role_summary}\n"
        f"Hard skills required: {hard_skills}\n"
        f"Soft skills required: {soft_skills}\n"
        f"Likely interview topics: {interview_topics}\n\n"
        f"=== CANDIDATE CV ===\n"
        f"{cv_block}\n\n"
        "=== TASK ===\n"
        "Execute the 6-step pipeline (silently), then output ONLY the final cover letter.\n"
        "Remember:\n"
        "- Infer the candidate's profile from their CV (do not assume any discipline).\n"
        "- Pick Top-2 or Top-3 experiences by relevance to the JD. Prefer work/internship "
        "over projects when both are strong matches.\n"
        "- Rewrite experiences in the candidate's own voice. Do not copy CV bullet points.\n"
        "- Map each experience to a specific JD requirement using concrete actions and outcomes.\n"
        "- Follow the 5-paragraph structure and all style constraints.\n"
        "- Run the self-check before returning."
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

