"""Career Co-pilot LangGraph Agent.

Defines a StateGraph that orchestrates the full pipeline:

  [START]
    → analyze_jd          (parse JD text into structured insights)
    → match_cv            (RAG: find CV passages matching JD, suggest rewrites)
    → refine_values       (value-refine selected CV chunks)
    → generate_cover_letter
    → [END]

The interview simulator is intentionally kept outside this graph because it is
interactive (multi-turn). The app invokes `InterviewSession` directly via
session state.

State schema
------------
  jd_text        : str   — raw JD input from the user
  cv_text        : str   — raw CV text (extracted from PDF)
  jd_analysis    : dict  — output of analyze_jd()
  match_results  : list  — output of match_cv_to_jd()
  refined_bullets: list[str] — value-refined CV bullets
  cover_letter   : str   — generated cover letter
  candidate_name : str   — optional; used in cover letter
  company_name   : str   — optional; used in cover letter
  error          : str   — populated if any node fails
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from career_copilot.jd_analyzer import analyze_jd
from career_copilot.cv_matcher import index_cv, match_cv_to_jd, compute_match_metrics
from career_copilot.value_refiner import refine_value
from career_copilot.cover_letter import generate_cover_letter


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    jd_text: str
    cv_text: str
    jd_analysis: dict[str, Any]
    match_results: list[dict[str, Any]]
    match_metrics: dict[str, Any]
    refined_bullets: list[str]
    cover_letter: str
    candidate_name: str
    company_name: str
    error: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_analyze_jd(state: AgentState) -> AgentState:
    try:
        analysis = analyze_jd(state["jd_text"])
        return {"jd_analysis": analysis}
    except Exception as exc:
        return {"error": f"JD analysis failed: {exc}"}


def node_match_cv(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}
    try:
        cv_text = state.get("cv_text", "")
        if cv_text.strip():
            index_cv(cv_text)
        matches = match_cv_to_jd(state["jd_analysis"])
        metrics = compute_match_metrics(state["jd_analysis"], cv_text, matches)
        return {"match_results": matches, "match_metrics": metrics}
    except Exception as exc:
        return {"error": f"CV matching failed: {exc}"}


def node_refine_values(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}
    matches = state.get("match_results", [])
    bullets: list[str] = []
    for m in matches[:5]:
        passage = m.get("suggested_rewrite") or m.get("original", "")
        if passage:
            try:
                bullets.append(refine_value(passage))
            except Exception:
                bullets.append(passage)
    return {"refined_bullets": bullets}


def node_generate_cover_letter(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}
    matches = state.get("match_results", [])
    if not matches:
        return {"cover_letter": ""}
    try:
        letter = generate_cover_letter(
            jd_analysis=state["jd_analysis"],
            match_results=matches,
            candidate_name=state.get("candidate_name", "the candidate"),
            company_name=state.get("company_name", "your company"),
        )
        return {"cover_letter": letter}
    except Exception as exc:
        return {"error": f"Cover letter generation failed: {exc}"}


# ---------------------------------------------------------------------------
# Router — skip cover letter if no CV was provided
# ---------------------------------------------------------------------------

def _route_after_match(state: AgentState) -> str:
    if state.get("error"):
        return END
    if state.get("match_results"):
        return "refine_values"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Build and compile the Career Co-pilot StateGraph."""
    g = StateGraph(AgentState)

    g.add_node("analyze_jd", node_analyze_jd)
    g.add_node("match_cv", node_match_cv)
    g.add_node("refine_values", node_refine_values)
    g.add_node("generate_cover_letter", node_generate_cover_letter)

    g.set_entry_point("analyze_jd")
    g.add_edge("analyze_jd", "match_cv")
    g.add_conditional_edges(
        "match_cv",
        _route_after_match,
        {"refine_values": "refine_values", END: END},
    )
    g.add_edge("refine_values", "generate_cover_letter")
    g.add_edge("generate_cover_letter", END)

    return g.compile()


# Singleton compiled graph — imported by app.py
graph = build_graph()


def run_pipeline(
    jd_text: str,
    cv_text: str = "",
    candidate_name: str = "the candidate",
    company_name: str = "your company",
) -> AgentState:
    """Convenience wrapper: run the full pipeline and return the final state."""
    initial: AgentState = {
        "jd_text": jd_text,
        "cv_text": cv_text,
        "candidate_name": candidate_name,
        "company_name": company_name,
    }
    return graph.invoke(initial)
