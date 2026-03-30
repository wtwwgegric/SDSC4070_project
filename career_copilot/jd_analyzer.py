"""JD Analyzer — extracts structured insights from a Job Description.

Returns a dict with:
  hard_skills       : list[str]  — technical skills / tools / languages
  soft_skills       : list[str]  — behavioural / interpersonal requirements
  jargon_decoded    : dict[str, str]  — buzzwords → plain-language meaning + interview tip
  interview_topics  : list[str]  — likely interview question themes
  summary           : str        — one-paragraph plain-English summary of the role
"""
import json
from typing import Any

from career_copilot.config import get_client, get_model


_SYSTEM = (
    "You are a senior career coach and talent acquisition expert. "
    "When given a Job Description (JD), you extract structured information accurately. "
    "Never fabricate requirements that are not stated or clearly implied in the JD."
)

_PROMPT_TEMPLATE = """Analyze the following Job Description and return a JSON object with exactly these keys:

- "hard_skills": list of technical skills, programming languages, frameworks, tools explicitly or clearly implied in the JD.
- "soft_skills": list of behavioural/interpersonal competencies required.
- "jargon_decoded": object where each key is a corporate buzzword or vague phrase found in the JD, and the value is an object with:
    - "meaning": plain-language explanation of what this really means in practice,
    - "interview_tip": a specific tip for the candidate when addressing this in an interview.
- "interview_topics": list of 5-8 likely interview question themes based on the JD.
- "summary": one concise paragraph summarising the role, team context, and key expectations.

Return ONLY valid JSON. No markdown fences, no extra text.

Job Description:
\"\"\"
{jd_text}
\"\"\"
"""


def analyze_jd(jd_text: str, model: str = None) -> dict[str, Any]:
    """Analyze a JD and return a structured insights dict.

    Raises ValueError if the LLM response cannot be parsed as JSON.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text must not be empty")

    client = get_client()
    model = model or get_model("gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _PROMPT_TEMPLATE.format(jd_text=jd_text.strip())},
        ],
        temperature=0.1,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON response: {raw[:300]}") from exc
