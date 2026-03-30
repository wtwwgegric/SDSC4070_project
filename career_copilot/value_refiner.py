from career_copilot.config import get_client, get_model


def refine_value(task_description: str, model: str = None) -> str:
    """Call OpenAI to turn a 'dirty work' description into a value-focused CV bullet.

    Keeps prompts conservative to avoid over-embellishment.
    """
    client = get_client()
    model = model or get_model("gpt-4o-mini")
    prompt = (
        "You are a seasoned career consultant. Transform the following seemingly mundane job description into commercially valuable CV entries."
        "Rules:\n"
        "1. Only use facts present in the original description, do not fabricate numbers or responsibilities.\n"
        "2. Output 1-2 sentences in English (suitable for a CV), with a quantifiable metric in parentheses if applicable.\n"
        f"Original description: {task_description}\n\nOutput:"

    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()
