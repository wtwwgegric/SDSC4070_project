"""Career Co-pilot — Streamlit App.

Tabs:
  1. JD Analyzer     — paste a JD → get structured insights + jargon decoder
  2. CV Matcher      — upload CV → RAG match against JD + rewrite suggestions
  3. Cover Letter    — auto-generate a grounded cover letter
  4. Interview Sim   — multi-turn interview with per-round rubric scoring
"""
import os
import json
import streamlit as st
import pandas as pd

from career_copilot.pdf_loader import load_pdf_from_bytes, chunk_text
from career_copilot.jd_analyzer import analyze_jd
from career_copilot.cv_matcher import index_cv, match_cv_to_jd
from career_copilot.value_refiner import refine_value
from career_copilot.cover_letter import generate_cover_letter
from career_copilot.interview_simulator import (
    InterviewSession, generate_self_intro_draft, prep_chat_response, prep_chat_summary
)
from career_copilot.serper import fetch_company_culture
from career_copilot.eval_metrics import (
    keyword_hit_rate,
    keyword_hit_rate_improvement,
    hallucination_check,
    rubric_summary,
)
from career_copilot.agent import run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Career Co-pilot", layout="wide", page_icon="🧭")

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_defaults = {
    "cv_text": "",
    "cv_indexed": False,
    "jd_analysis": None,
    "match_results": None,
    "cover_letter": "",
    "interview_session": None,
    "interview_scores": [],
    "interview_history": [],   # list of {question, answer, score}
    "interview_current_q": "",
    "interview_phase": "idle", # idle | self_intro | awaiting_answer | reviewing_feedback | done
    "interview_self_intro": "",
    "interview_mode": "prep",          # "prep" | "test"
    "prep_chat_history": [],            # list of {role, content}
    "prep_chat_summary": "",
    "candidate_name": "",
    "company_name": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar — CV upload + company lookup
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 Career Co-pilot")
    st.caption("AI-powered job search assistant")
    st.divider()

    st.subheader("1. Upload your CV")
    uploaded_cv = st.file_uploader("PDF only", type=["pdf"], key="cv_uploader")
    if uploaded_cv:
        raw_bytes = uploaded_cv.read()
        with st.spinner("Extracting text from PDF…"):
            extracted = load_pdf_from_bytes(raw_bytes)
            st.session_state["cv_text"] = extracted
            st.session_state["cv_indexed"] = False
        if extracted:
            st.success(f"Extracted {len(extracted)} characters")
        else:
            st.error(
                "Could not extract text from this PDF. "
                "It may be a scanned/image-only file. "
                "Please use a text-based PDF or copy-paste your CV text manually."
            )

    if st.session_state["cv_text"]:
        st.caption(f"CV loaded ✅ ({len(st.session_state['cv_text'])} chars)")

    st.divider()
    st.subheader("2. Your details")
    st.session_state["candidate_name"] = st.text_input(
        "Your name", value=st.session_state["candidate_name"], placeholder="Jane Doe"
    )
    st.session_state["company_name"] = st.text_input(
        "Target company", value=st.session_state["company_name"], placeholder="Google"
    )

    st.divider()
    st.subheader("3. ⚡ Full Pipeline (one-click)")
    st.caption("Runs JD analysis → CV matching → cover letter in sequence.")
    if st.button("🚀 Run Full Pipeline", use_container_width=True):
        jd_text = st.session_state.get("jd_input_text", "").strip()
        cv_text = st.session_state.get("cv_text", "").strip()
        if not jd_text:
            st.warning("Paste a JD in Tab 1 first.")
        elif not cv_text:
            st.warning("Upload your CV first.")
        else:
            with st.spinner("Running full pipeline… (this may take ~30 s)"):
                try:
                    result = run_pipeline(
                        jd_text=jd_text,
                        cv_text=cv_text,
                        candidate_name=st.session_state.get("candidate_name") or "the candidate",
                        company_name=st.session_state.get("company_name") or "your company",
                    )
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.session_state["jd_analysis"] = result.get("jd_analysis")
                        st.session_state["match_results"] = result.get("match_results")
                        st.session_state["cv_indexed"] = True
                        st.session_state["cover_letter"] = result.get("cover_letter", "")
                        st.success("Pipeline complete! Check all tabs.")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

    st.divider()
    st.subheader("4. Company culture lookup")
    company_query = st.text_input("Company name", key="culture_company")
    if st.button("🔍 Fetch culture hints") and company_query:
        try:
            with st.spinner("Querying Serper.dev…"):
                hits = fetch_company_culture(company_query, num_results=5)
            for h in hits:
                if h.get("title"):
                    st.markdown(f"**{h['title']}**")
                if h.get("snippet"):
                    st.write(h["snippet"])
                if h.get("link"):
                    st.caption(h["link"])
                st.write("---")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.caption("Set `OPENAI_API_KEY` (and optionally `SERPER_API_KEY`) in your environment.")

# ---------------------------------------------------------------------------
# Main area — 4 tabs
# ---------------------------------------------------------------------------
tab_jd, tab_cv, tab_cl, tab_sim = st.tabs(
    ["📋 JD Analyzer", "🎯 CV Matcher", "✉️ Cover Letter", "🎤 Interview Sim"]
)

# ============================================================
# Tab 1 — JD Analyzer
# ============================================================
with tab_jd:
    st.header("Job Description Analyzer")
    st.markdown(
        "Paste a JD below. The agent will extract hard skills, soft skills, "
        "decode corporate jargon, and suggest interview preparation topics."
    )

    jd_input = st.text_area(
        "Paste Job Description here",
        height=250,
        placeholder="e.g. We are looking for a Data Engineer to join our fast-paced team…",
        key="jd_input_text",
    )

    if st.button("🔍 Analyze JD", type="primary"):
        if not jd_input.strip():
            st.warning("Please paste a JD first.")
        else:
            with st.spinner("Analyzing JD…"):
                try:
                    result = analyze_jd(jd_input)
                    st.session_state["jd_analysis"] = result
                    # Reset downstream state when JD changes
                    st.session_state["match_results"] = None
                    st.session_state["cover_letter"] = ""
                    st.session_state["cv_indexed"] = False
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    analysis = st.session_state.get("jd_analysis")
    if analysis:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔧 Hard Skills")
            for s in analysis.get("hard_skills", []):
                st.markdown(f"- `{s}`")
            st.subheader("🤝 Soft Skills")
            for s in analysis.get("soft_skills", []):
                st.markdown(f"- {s}")

        with col2:
            st.subheader("🗣️ Interview Topics")
            for t in analysis.get("interview_topics", []):
                st.markdown(f"- {t}")

        st.divider()
        st.subheader("🕵️ Jargon Decoder")
        jargon = analysis.get("jargon_decoded", {})
        if jargon:
            for phrase, info in jargon.items():
                with st.expander(f'**"{phrase}"**'):
                    st.markdown(f"**What it really means:** {info.get('meaning', '')}")
                    st.info(f"💡 Interview tip: {info.get('interview_tip', '')}")
        else:
            st.caption("No jargon detected.")

        st.divider()
        st.subheader("📝 Role Summary")
        st.write(analysis.get("summary", ""))

# ============================================================
# Tab 2 — CV Matcher
# ============================================================
with tab_cv:
    st.header("CV ↔ JD Matcher")
    st.markdown(
        "Indexes your CV into a vector store, then retrieves the passages most "
        "relevant to each JD requirement and suggests improved rewrites."
    )

    analysis = st.session_state.get("jd_analysis")
    cv_text = st.session_state.get("cv_text", "")

    if not analysis:
        st.info("➡️ Analyze a JD first (Tab 1).")
    elif not cv_text:
        st.info("➡️ Upload your CV in the sidebar.")
    else:
        # Value Refiner (single chunk)
        st.subheader("✨ Value Refiner")
        st.caption("Transform a 'dirty work' description into a polished CV bullet.")
        chunks = chunk_text(cv_text, chunk_size=1200, overlap=200)
        def _chunk_label(i: int) -> str:
            # Show the first non-empty line of the chunk for a meaningful preview
            first_line = next((l.strip() for l in chunks[i].split("\n") if l.strip()), "")
            return f"Chunk {i+1}: {first_line[:100]}{'…' if len(first_line) > 100 else ''}"

        sel = st.selectbox(
            "Select a CV chunk to refine",
            options=list(range(min(15, len(chunks)))),
            format_func=_chunk_label,
        )
        manual_input = st.text_area("Or type a description manually", key="manual_refine")
        to_refine = manual_input.strip() or (chunks[sel] if chunks else "")

        if st.button("✨ Refine this bullet"):
            with st.spinner("Refining…"):
                try:
                    refined = refine_value(to_refine)
                    st.success("Refined:")
                    st.write(refined)
                except Exception as e:
                    st.error(str(e))

        st.divider()

        # Show all CV chunks so user can verify quality
        with st.expander(f"🔍 Preview all CV chunks ({len(chunks)} total)"):
            for ci, ch in enumerate(chunks):
                st.markdown(f"**Chunk {ci+1}**")
                st.text(ch)
                st.write("---")

        st.divider()

        # Full RAG match
        st.subheader("🎯 Full JD ↔ CV Match")
        if st.button("🚀 Run CV Matching", type="primary"):
            with st.spinner("Indexing CV & running RAG match… (this calls the OpenAI Embeddings API)"):
                try:
                    if not st.session_state["cv_indexed"]:
                        index_cv(cv_text)
                        st.session_state["cv_indexed"] = True
                    matches = match_cv_to_jd(analysis)
                    st.session_state["match_results"] = matches
                except Exception as e:
                    st.error(f"Matching failed: {e}")

        matches = st.session_state.get("match_results")
        if matches:
            st.success(f"Found {len(matches)} relevant passages.")

            # Keyword hit rate
            hit_before = keyword_hit_rate(analysis, cv_text)
            after_text = " ".join(m.get("suggested_rewrite", "") for m in matches)
            hit_after = keyword_hit_rate(analysis, cv_text + " " + after_text)
            c1, c2, c3 = st.columns(3)
            c1.metric("Keyword Hit Rate (before)", f"{hit_before:.1%}")
            c2.metric("Keyword Hit Rate (after rewrites)", f"{hit_after:.1%}")
            c3.metric("Improvement", f"+{hit_after - hit_before:.1%}")

            st.divider()
            for i, m in enumerate(matches, 1):
                with st.expander(f"Match {i}: `{m['keyword']}`  (similarity score: {1 - m['score']:.2f})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Original CV passage**")
                        st.text(m["original"])
                    with col_b:
                        st.markdown("**Suggested rewrite**")
                        st.success(m["suggested_rewrite"])

# ============================================================
# Tab 3 — Cover Letter
# ============================================================
with tab_cl:
    st.header("Cover Letter Generator")
    st.markdown(
        "Generates a **grounded** cover letter: every claim is traceable to your CV. "
        "Requires JD analysis (Tab 1) + CV matching (Tab 2) to be completed first."
    )

    analysis = st.session_state.get("jd_analysis")
    matches = st.session_state.get("match_results")

    if not analysis:
        st.info("➡️ Analyze a JD first (Tab 1).")
    elif not matches:
        st.info("➡️ Run CV Matching first (Tab 2).")
    else:
        candidate = st.session_state.get("candidate_name") or "the candidate"
        company = st.session_state.get("company_name") or "your company"

        if st.button("✉️ Generate Cover Letter", type="primary"):
            with st.spinner("Writing cover letter…"):
                try:
                    letter = generate_cover_letter(
                        jd_analysis=analysis,
                        match_results=matches,
                        candidate_name=candidate,
                        company_name=company,
                    )
                    st.session_state["cover_letter"] = letter
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if st.session_state.get("cover_letter"):
            st.subheader("📄 Your Cover Letter")
            st.write(st.session_state["cover_letter"])

            # Hallucination / grounding check
            cv_text_for_check = st.session_state.get("cv_text", "")
            if cv_text_for_check:
                hc = hallucination_check(cv_text_for_check, st.session_state["cover_letter"])
                ratio = hc["traceability_ratio"]
                hcol1, hcol2, hcol3 = st.columns(3)
                hcol1.metric("📎 Grounding score", f"{ratio:.0%}",
                             help="Fraction of phrases traceable back to your CV. Higher = less hallucination.")
                hcol2.metric("Traceable phrases", hc["traceable"])
                hcol3.metric("Total phrases checked", hc["total_phrases"])
                if hc.get("untraceable_samples"):
                    with st.expander("⚠️ Phrases not directly matched in CV (review manually)"):
                        for s in hc["untraceable_samples"]:
                            st.caption(f"- {s}")

            st.download_button(
                "⬇️ Download as .txt",
                data=st.session_state["cover_letter"],
                file_name="cover_letter.txt",
                mime="text/plain",
            )

# ============================================================
# Tab 4 — Interview Simulator
# ============================================================
with tab_sim:
    st.header("Interview Simulator")
    st.markdown(
        "Practice a **multi-turn interview** based on the JD and your CV. "
        "Each answer is scored on Technical Accuracy, Completeness, and Clarity (1–5)."
    )

    analysis = st.session_state.get("jd_analysis")
    matches = st.session_state.get("match_results", [])
    phase = st.session_state["interview_phase"]
    session: InterviewSession | None = st.session_state.get("interview_session")

    if not analysis:
        st.info("➡️ Analyze a JD first (Tab 1).")
    else:
        # ── Mode toggle ──────────────────────────────────────────────
        mode = st.radio(
            "Interview mode",
            options=["🏋️ Prep Chat", "📝 Mock Test"],
            index=0 if st.session_state["interview_mode"] == "prep" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state["interview_mode"] = "prep" if mode == "🏋️ Prep Chat" else "test"
        st.divider()

        # ── PREP CHAT MODE ────────────────────────────────────────────
        if st.session_state["interview_mode"] == "prep":
            st.markdown(
                "Chat freely with your AI career coach. Ask about potential questions, "
                "get STAR talking-point suggestions, or share a draft answer for feedback."
            )
            st.caption(
                "💡 Try: *'What questions might they ask about my Python experience?'* "
                "or *'How should I answer a weakness question?'* "
                "or *'Is this a good answer: …'*"
            )

            # Self-intro reference
            self_intro_txt = st.session_state.get("interview_self_intro", "")
            if not self_intro_txt:
                cv_for_intro = st.session_state.get("cv_text", "")
                col_gen, col_skip = st.columns([1, 3])
                with col_gen:
                    if st.button("✨ Generate self-intro first", disabled=not cv_for_intro):
                        with st.spinner("Drafting self-introduction…"):
                            try:
                                draft = generate_self_intro_draft(cv_for_intro, analysis)
                                st.session_state["interview_self_intro"] = draft
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not generate draft: {e}")
                if not cv_for_intro:
                    st.caption("Upload CV in the sidebar to enable auto self-intro.")
            else:
                with st.expander("📌 Your Self-Introduction (click to read/edit)", expanded=False):
                    edited = st.text_area(
                        "Edit your intro",
                        value=self_intro_txt,
                        height=120,
                        key="prep_intro_edit",
                        label_visibility="collapsed",
                    )
                    if edited != self_intro_txt:
                        st.session_state["interview_self_intro"] = edited

            st.divider()

            # Chat history display
            prep_history = st.session_state["prep_chat_history"]
            for msg in prep_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            user_msg = st.chat_input("Ask your career coach…")
            if user_msg:
                st.session_state["prep_chat_history"].append(
                    {"role": "user", "content": user_msg}
                )
                with st.chat_message("user"):
                    st.markdown(user_msg)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            reply = prep_chat_response(
                                user_message=user_msg,
                                chat_history=prep_history[:-1],  # exclude the one just appended
                                jd_analysis=analysis,
                                cv_text=st.session_state.get("cv_text", ""),
                                self_intro=st.session_state.get("interview_self_intro", ""),
                            )
                        except Exception as e:
                            reply = f"⚠️ Error: {e}"
                    st.markdown(reply)
                st.session_state["prep_chat_history"].append(
                    {"role": "assistant", "content": reply}
                )

            if prep_history:
                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if st.button("📋 Summarise this session"):
                        with st.spinner("Summarising…"):
                            try:
                                summary = prep_chat_summary(
                                    chat_history=prep_history,
                                    jd_analysis=analysis,
                                )
                                st.session_state["prep_chat_summary"] = summary
                            except Exception as e:
                                st.error(f"Could not summarise: {e}")
                with btn_col2:
                    if st.button("🗑️ Clear chat"):
                        st.session_state["prep_chat_history"] = []
                        st.session_state["prep_chat_summary"] = ""
                        st.rerun()

                summary_text = st.session_state.get("prep_chat_summary", "")
                if summary_text:
                    with st.expander("📋 Session Summary (save this for your notes)", expanded=True):
                        st.markdown(summary_text)

        # ── MOCK TEST MODE ────────────────────────────────────────────
        else:
            col_start, col_reset = st.columns([1, 1])
            with col_start:
                if phase == "idle" and st.button("▶ Start Interview", type="primary"):
                    st.session_state["interview_phase"] = "self_intro"
                    st.session_state["interview_self_intro"] = ""
                    st.rerun()
            with col_reset:
                if phase != "idle" and st.button("🔄 Reset Interview"):
                    st.session_state["interview_session"] = None
                    st.session_state["interview_scores"] = []
                    st.session_state["interview_history"] = []
                    st.session_state["interview_current_q"] = ""
                    st.session_state["interview_self_intro"] = ""
                    st.session_state["interview_phase"] = "idle"
                    st.rerun()

            # ── Self-introduction phase ────────────────────────────────────────
            if phase == "self_intro":
                st.subheader("👋 Step 1 — Prepare your self-introduction")
                st.markdown(
                    "A strong self-intro sets the tone. Generate a draft below, "
                    "then edit it to match your own voice before starting the interview."
                )
                cv_text_for_intro = st.session_state.get("cv_text", "")

                col_gen, _ = st.columns([1, 2])
                with col_gen:
                    if st.button("✨ Generate draft", disabled=not cv_text_for_intro):
                        with st.spinner("Drafting your self-introduction…"):
                            try:
                                draft = generate_self_intro_draft(cv_text_for_intro, analysis)
                                st.session_state["interview_self_intro"] = draft
                            except Exception as e:
                                st.error(f"Could not generate draft: {e}")
                    if not cv_text_for_intro:
                        st.caption("Upload your CV in the sidebar to enable auto-draft.")

                intro_text = st.text_area(
                    "Your self-introduction (edit freely)",
                    value=st.session_state["interview_self_intro"],
                    height=160,
                    placeholder="Hi, I'm … I've been working on … and I'm excited about this role because …",
                    key="self_intro_textarea",
                )
                st.session_state["interview_self_intro"] = intro_text

                if st.button("▶ Start Interview with this introduction", type="primary"):
                    sess = InterviewSession(
                        jd_analysis=analysis,
                        match_results=matches or [],
                        self_intro=intro_text.strip(),
                    )
                    st.session_state["interview_session"] = sess
                    st.session_state["interview_scores"] = []
                    st.session_state["interview_history"] = []
                    q = sess.next_question()
                    st.session_state["interview_current_q"] = q
                    st.session_state["interview_phase"] = "awaiting_answer"
                    st.rerun()

            # ── Persistent self-intro reference panel ───────────────────
            if phase in ("awaiting_answer", "reviewing_feedback", "done"):
                self_intro_text = st.session_state.get("interview_self_intro", "")
                if self_intro_text:
                    with st.expander("📌 Your Self-Introduction (for reference)", expanded=False):
                        st.info(self_intro_text)
                    st.divider()

            # ── Chat history ────────────────────────────────────────────
            history = st.session_state["interview_history"]
            if history:
                st.subheader("🗒️ Interview History")
                for i, entry in enumerate(history):
                    with st.container(border=True):
                        st.markdown(f"**Q{i+1}:** {entry['question']}")
                        st.markdown(f"**Your answer:** {entry['answer']}")
                        sc = entry["score"]
                        with st.expander(f"📊 Score: {sc.overall:.1f}/5 — click to expand"):
                            hc1, hc2, hc3, hc4 = st.columns(4)
                            hc1.metric("Technical", f"{sc.technical_accuracy}/5")
                            hc2.metric("Completeness", f"{sc.completeness}/5")
                            hc3.metric("Clarity", f"{sc.clarity}/5")
                            hc4.metric("Overall", f"{sc.overall:.1f}/5")
                            st.caption(f"💬 {sc.feedback}")
                st.divider()

            # ── Awaiting answer ──────────────────────────────────────────
            if phase == "awaiting_answer" and session:
                total = session._max_questions
                done_count = len(history)
                st.subheader(f"🎙️ Question {done_count + 1} of {total}")
                st.info(st.session_state["interview_current_q"])
                user_answer = st.text_area("Your answer", key="user_answer_input", height=150,
                                           placeholder="Type your answer here…")
                if st.button("Submit Answer", type="primary"):
                    if not user_answer.strip():
                        st.warning("Please type an answer before submitting.")
                    else:
                        with st.spinner("Evaluating your answer…"):
                            score = session.answer(user_answer)
                        entry = {
                            "question": st.session_state["interview_current_q"],
                            "answer": user_answer,
                            "score": score,
                        }
                        st.session_state["interview_history"].append(entry)
                        st.session_state["interview_scores"].append(score)
                        st.session_state["interview_phase"] = "reviewing_feedback"
                        st.rerun()

            # ── Reviewing feedback ───────────────────────────────────────
            elif phase == "reviewing_feedback" and session and history:
                last = history[-1]
                sc = last["score"]
                rounds_done = len(history)
                total = session._max_questions

                st.subheader(f"📊 Feedback — Question {rounds_done} of {total}")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Technical", f"{sc.technical_accuracy}/5")
                fc2.metric("Completeness", f"{sc.completeness}/5")
                fc3.metric("Clarity", f"{sc.clarity}/5")
                fc4.metric("Overall", f"{sc.overall:.1f}/5")
                st.info(f"💬 {sc.feedback}")

                if rounds_done >= total:
                    if st.button("🏁 View Final Report", type="primary"):
                        st.session_state["interview_phase"] = "done"
                        st.rerun()
                else:
                    if st.button("➡️ Next Question", type="primary"):
                        next_q = session.next_question()
                        st.session_state["interview_current_q"] = next_q
                        st.session_state["interview_phase"] = "awaiting_answer"
                        st.rerun()

            # ── Final report ─────────────────────────────────────────────
            elif phase == "done" and session:
                st.subheader("🏁 Interview Complete — Final Report")
                with st.spinner("Generating closing assessment…"):
                    report = session.final_report()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Technical", f"{report['avg_technical_accuracy']:.1f}/5")
                col2.metric("Avg Completeness", f"{report['avg_completeness']:.1f}/5")
                col3.metric("Avg Clarity", f"{report['avg_clarity']:.1f}/5")
                col4.metric("Avg Overall", f"{report['avg_overall']:.1f}/5")

                st.divider()
                st.subheader("📈 Score Progression")
                scores = st.session_state["interview_scores"]
                if scores:
                    chart_data = pd.DataFrame(rubric_summary(scores)).set_index("Round")
                    st.line_chart(chart_data)

                st.divider()
                st.subheader("💬 Closing Assessment")
                st.write(report.get("closing_assessment", ""))

                with st.expander("📊 Per-round breakdown"):
                    for r in report.get("per_round", []):
                        st.markdown(
                            f"**Round {r['round']}** — "
                            f"Technical: {r['technical_accuracy']}, "
                            f"Completeness: {r['completeness']}, "
                            f"Clarity: {r['clarity']}, "
                            f"Overall: {r['overall']}"
                        )
                        st.caption(f"Q: {r['question']}")
                        st.caption(f"Feedback: {r['feedback']}")
                        st.write("---")
