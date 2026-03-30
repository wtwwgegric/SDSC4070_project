from career_copilot.config import get_client, get_model


def refine_value(task_description: str, model: str = None) -> str:
    """Call OpenAI to turn a 'dirty work' description into a value-focused CV bullet.

    Keeps prompts conservative to avoid over-embellishment.
    """
    client = get_client()
    model = model or get_model("gpt-4o-mini")
    prompt = (
        "你是一位資深職涯顧問。將以下看似平凡的工作描述，轉化為具備商業價值的 CV 條目。"
        "規則：\n"
        "1. 只能使用原描述中實際存在的事實，不可捏造數字或職責。\n"
        "2. 輸出 1-2 句英文（適合 CV），句末用括號補充一個可量化指標（若可推算）。\n"
        "3. 同時輸出一行中文摘要。\n\n"
        f"原始描述：{task_description}\n\n輸出："
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()
