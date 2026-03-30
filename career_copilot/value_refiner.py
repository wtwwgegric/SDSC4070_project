import os
import json
import openai


def _ensure_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment")
    openai.api_key = key


def refine_value(task_description: str, model: str = None) -> str:
    """Call OpenAI to turn a 'dirty work' description into a value-focused bullet.

    Returns a text response (LLM output). Keep prompts conservative to avoid over-embellishment.
    """
    _ensure_api_key()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "把這項看似無聊的工作轉化為具備商業價值的描述。" 
        "輸出一個簡潔的 1-2 句話描述，並在最後用括號給出可量化的指標（如果可行）。"
        f"\n\n原始描述：{task_description}\n\n輸出："
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    text = resp.choices[0].message.get("content", "").strip()
    # try to keep output JSON-friendly if user wants to parse later
    try:
        # if model returned JSON, keep as-is; otherwise wrap
        json.loads(text)
        return text
    except Exception:
        return text
