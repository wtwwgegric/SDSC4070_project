"""CV Matcher — uses RAG to find CV passages that match a JD and suggests rewrites.

Workflow:
1. Index CV chunks into an in-memory ChromaDB collection (or reuse existing).
2. Query the collection with each JD hard skill / topic.
3. For each matched passage, ask the LLM to suggest a rewritten version that better
   targets the JD keyword — without fabricating new facts.

Returns a list of MatchResult dicts, each containing:
  keyword         : str   — the JD keyword used as query
  original        : str   — the best-matching CV passage
  suggested_rewrite: str  — LLM-suggested improved version
  score           : float — cosine distance (lower = more similar)
"""
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
    """
    keywords: list[str] = jd_analysis.get("hard_skills", []) + jd_analysis.get("interview_topics", [])
    if not keywords:
        return []

    client = get_client()
    model = model or get_model("gpt-4o-mini")
    results: list[dict[str, Any]] = []

    seen_passages: set[str] = set()

    for keyword in keywords[:10]:  # cap to keep API costs low
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

        # Skip duplicates (same passage matched by different keywords)
        if best_doc in seen_passages:
            continue
        seen_passages.add(best_doc)

        # Ask LLM for a better rewrite
        rewrite_prompt = (
            f"CV passage:\n\"{best_doc}\"\n\n"
            f"Target JD keyword: \"{keyword}\"\n\n"
            "Provide an improved CV bullet point that highlights this keyword. "
            "Only use facts from the passage above."
        )
        rewrite_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": rewrite_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        rewrite = rewrite_resp.choices[0].message.content.strip()

        results.append({
            "keyword": keyword,
            "original": best_doc.strip(),
            "suggested_rewrite": rewrite,
            "score": round(best_dist, 4),
        })

    return results
