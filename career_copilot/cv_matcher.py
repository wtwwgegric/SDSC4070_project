"""CV Matcher — uses RAG to find CV passages that match a JD and suggests rewrites.

Workflow:
1. Index CV chunks into an in-memory ChromaDB collection (or reuse existing).
2. Query the collection with each JD hard skill / topic.
3. For each matched passage, ask the LLM to suggest a rewritten version that better
   targets the JD keyword — without fabricating new facts.
4. Compute structured match metrics (skill/experience/requirement scores).
5. Provide a chatbot interface for conversational match insights.

Returns a list of MatchResult dicts, each containing:
  keyword         : str   — the JD keyword used as query
  original        : str   — the best-matching CV passage
  suggested_rewrite: str  — LLM-suggested improved version
  score           : float — cosine distance (lower = more similar)
"""
from __future__ import annotations

import json
from typing import Any

from career_copilot.config import get_client, get_model
from career_copilot.rag import create_collection, query_collection
from career_copilot.pdf_loader import chunk_text


_COLLECTION_NAME = "cv_chunks"
_REWRITE_SYSTEM = (
    "You are a professional CV editor. "
    "Given a CV passage and a target JD keyword, rewrite the passage so it better "
    "highlights relevance to that keyword. "
    "RULES: Use only facts present in the original passage. Do NOT invent numbers, "
    "projects, or responsibilities. Keep the rewrite under 3 sentences."
)


def index_cv(cv_text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Chunk CV text and store it in an ephemeral ChromaDB collection.

    Returns the list of chunks that were indexed.
    """
    chunks = chunk_text(cv_text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("CV text is empty — nothing to index")
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    create_collection(
        name=_COLLECTION_NAME,
        texts=chunks,
        metadatas=metadatas,
        persist_directory=None,  # ephemeral — lives for this session
    )
    return chunks


def match_cv_to_jd(
    jd_analysis: dict[str, Any],
    top_k: int = 3,
    model: str = None,
) -> list[dict[str, Any]]:
    """Query indexed CV against JD hard skills and return match results with rewrites.

    `jd_analysis` should be the output of `analyze_jd()`.
    Requires `index_cv()` to have been called first.

    Rewrites are batched into a single LLM call to minimize latency.
    """
    keywords: list[str] = jd_analysis.get("hard_skills", []) + jd_analysis.get("interview_topics", [])
    if not keywords:
        return []

    results: list[dict[str, Any]] = []
    seen_passages: set[str] = set()

    # Phase 1: RAG retrieval only (fast — just vector queries)
    for keyword in keywords[:10]:
        try:
            res = query_collection(_COLLECTION_NAME, query=keyword, k=top_k)
        except Exception:
            continue

        docs = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]

        if not docs:
            continue

        best_doc = docs[0]
        best_dist = distances[0] if distances else 1.0

        if best_doc in seen_passages:
            continue
        seen_passages.add(best_doc)

        results.append({
            "keyword": keyword,
            "original": best_doc.strip(),
            "suggested_rewrite": "",
            "score": round(best_dist, 4),
        })

    # Phase 2: Batch all rewrites in one LLM call (instead of N separate calls)
    if results:
        client = get_client()
        model = model or get_model("qwen-plus")

        batch_prompt_parts = []
        for i, m in enumerate(results):
            batch_prompt_parts.append(
                f"[{i+1}] Keyword: \"{m['keyword']}\"\n"
                f"CV passage: \"{m['original'][:300]}\"\n"
            )
        batch_prompt = (
            "For each numbered item below, provide an improved CV bullet point "
            "that highlights the target keyword. Use ONLY facts from the passage. "
            "Keep each rewrite under 3 sentences.\n"
            "Return your answer as a numbered list (1. … 2. … etc.)\n\n"
            + "\n".join(batch_prompt_parts)
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": batch_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            raw = resp.choices[0].message.content.strip()
            # Parse numbered list: "1. ..." "2. ..." etc.
            import re as _re
            parts = _re.split(r'\n(?=\d+\.\s)', raw)
            for part in parts:
                m_num = _re.match(r'(\d+)\.\s*(.*)', part, _re.DOTALL)
                if m_num:
                    idx = int(m_num.group(1)) - 1
                    if 0 <= idx < len(results):
                        results[idx]["suggested_rewrite"] = m_num.group(2).strip()
        except Exception:
            pass  # rewrites are optional — metrics & chat still work

    return results


# ---------------------------------------------------------------------------
# Structured match metrics
# ---------------------------------------------------------------------------

def compute_match_metrics(
    jd_analysis: dict[str, Any],
    cv_text: str,
    match_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute structured match metrics between CV and JD.

    Returns a dict with:
      skill_match_score       : float — fraction of hard skills found in CV
      experience_match_score  : float — avg RAG similarity of top matches (0-1)
      requirement_match_score : float — fraction of interview topics covered
      overall_match_score     : float — weighted average
      matched_skills          : list[str]
      missing_skills          : list[str]
      matched_skills_count    : int
      missing_skills_count    : int
      matched_topics          : list[str]
      missing_topics          : list[str]
    """
    cv_lower = cv_text.lower()

    # --- Skill matching ---
    hard_skills = [s.strip() for s in jd_analysis.get("hard_skills", [])]
    matched_skills = [s for s in hard_skills if s.lower() in cv_lower]
    missing_skills = [s for s in hard_skills if s.lower() not in cv_lower]
    skill_score = len(matched_skills) / len(hard_skills) if hard_skills else 0.0

    # --- Experience matching (RAG similarity) ---
    if match_results:
        # score is cosine distance — lower = better; convert to similarity
        similarities = [max(0.0, 1.0 - m["score"]) for m in match_results]
        experience_score = sum(similarities) / len(similarities)
    else:
        experience_score = 0.0

    # --- Requirement / topic coverage ---
    topics = [t.strip() for t in jd_analysis.get("interview_topics", [])]
    matched_topics = [t for t in topics if t.lower() in cv_lower]
    missing_topics = [t for t in topics if t.lower() not in cv_lower]
    requirement_score = len(matched_topics) / len(topics) if topics else 0.0

    # --- Overall ---
    overall = round(0.45 * skill_score + 0.30 * experience_score + 0.25 * requirement_score, 4)

    return {
        "skill_match_score": round(skill_score, 4),
        "experience_match_score": round(experience_score, 4),
        "requirement_match_score": round(requirement_score, 4),
        "overall_match_score": overall,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_skills_count": len(matched_skills),
        "missing_skills_count": len(missing_skills),
        "matched_topics": matched_topics,
        "missing_topics": missing_topics,
    }


# ---------------------------------------------------------------------------
# Chatbot for match insights
# ---------------------------------------------------------------------------

_MATCH_CHAT_SYSTEM = """\
You are a career advisor AI embedded in a CV-JD matching tool.

You have access to:
1. The full JD analysis (hard skills, soft skills, interview topics, summary).
2. The candidate's full CV text.
3. Structured match metrics (skill/experience/requirement scores, matched & missing skills).
4. RAG match results (which CV passages matched which JD keywords, with similarity scores).

Your job is to answer the user's questions about their CV-JD fit in a **direct, actionable** way.
Common questions include:
- Which of my skills match perfectly?
- Which experiences are relevant to this role?
- What hard skills am I missing?
- How should I modify my CV to better fit this JD?
- What's my strongest selling point for this role?

Rules:
- Be specific: cite actual skill names, project names, experience details from the CV.
- Be honest about gaps — don't sugar-coat missing skills.
- When suggesting CV improvements, give concrete, actionable advice (not generic platitudes).
- Keep responses concise but thorough.
- Use bullet points and bold text for readability.
- Respond in the same language the user writes in.
"""


def match_chat_response(
    user_message: str,
    jd_analysis: dict[str, Any],
    cv_text: str,
    match_metrics: dict[str, Any],
    match_results: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
    model: str | None = None,
) -> str:
    """Generate a chatbot response about CV-JD match insights."""
    client = get_client()
    model = model or get_model("qwen-plus")

    # Build context block
    context = (
        "=== JD ANALYSIS ===\n"
        f"{json.dumps(jd_analysis, ensure_ascii=False, indent=2)}\n\n"
        "=== MATCH METRICS ===\n"
        f"{json.dumps(match_metrics, ensure_ascii=False, indent=2)}\n\n"
        "=== TOP RAG MATCHES ===\n"
    )
    for m in match_results[:8]:
        context += (
            f"- Keyword: {m['keyword']} | Similarity: {1 - m['score']:.2f}\n"
            f"  CV passage: {m['original'][:200]}…\n"
        )
    context += f"\n=== FULL CV ===\n{cv_text[:4000]}\n"

    messages = [
        {"role": "system", "content": _MATCH_CHAT_SYSTEM},
        {"role": "user", "content": f"[CONTEXT — do not print this to the user]\n{context}"},
    ]
    # Append chat history
    for msg in (chat_history or []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()
