"""Interview Simulator — multi-turn interviewer with typed questions and rubric scoring.

Usage
-----
session = InterviewSession(jd_analysis, match_results, interview_style="1-to-1 interview")
question = session.next_question()          # get the first question
score    = session.answer(user_reply)       # submit answer -> get RubricScore
question = session.next_question()          # get next question from the interview plan
...
report   = session.final_report()           # summary after the interview

RubricScore fields (each 1-5):
  technical_accuracy  — correctness / relevance / judgment for this question type
  completeness        — depth of the answer (covers key points)
  clarity             — logical flow and communication quality
  overall             — weighted average
  feedback            — short qualitative comment
  question_type       — behavioural / technical / motivational / etc.
"""
import json
from dataclasses import dataclass, field
from typing import Any

from career_copilot.config import get_client, get_model


QUESTION_TYPE_LABELS = {
    "behavioral": "Behavioral",
    "motivational": "Motivational",
    "situational": "Situational",
    "resume_deep_dive": "Resume Deep Dive",
    "technical": "Technical",
    "technical_reasoning": "Technical Reasoning",
}


INTERVIEW_PLANS = {
    "1-to-1 interview": [
        {
            "type": "behavioral",
            "instruction": (
                "Ask a behavioral STAR-style question about past teamwork, ownership, "
                "problem-solving, or resilience. You want to see a concrete example with "
                "context, the candidate's specific actions, and a measurable or observable result."
            ),
        },
        {
            "type": "motivational",
            "instruction": (
                "Ask about the candidate's motivation for this role and company. "
                "You are looking for a coherent narrative arc: what past experience shaped their interest, "
                "how they arrived at this career direction, and what they hope to achieve in this position. "
                "Genuine self-awareness matters more than rehearsed enthusiasm."
            ),
        },
        {
            "type": "situational",
            "instruction": (
                "Ask how the candidate would handle ambiguity, conflicting priorities, "
                "or stakeholder pressure in this role. You want to see structured reasoning "
                "and practical decision-making, not textbook answers."
            ),
        },
        {
            "type": "resume_deep_dive",
            "instruction": (
                "Pick one project, internship, or achievement from the candidate's background "
                "and ask them to walk you through it in detail—their specific contribution, "
                "the challenges they faced, and what they would do differently in hindsight."
            ),
        },
        {
            "type": "behavioral",
            "instruction": (
                "Ask another behavioral question focusing on collaboration, communication, "
                "or learning from failure. Do not repeat earlier themes. You want to see "
                "self-reflection and growth, not just a success story."
            ),
        },
    ],
    "Technical interview": [
        {
            "type": "technical",
            "instruction": (
                "Ask a practical technical question tied to the role's core hard skills. "
                "Focus on applied reasoning and trade-offs, not trivia or syntax recall. "
                "You want to assess whether the candidate can think through a real problem."
            ),
        },
        {
            "type": "resume_deep_dive",
            "instruction": (
                "Ask the candidate to explain a technically relevant project from their background "
                "in depth—architecture decisions, challenges encountered, and how they measured success."
            ),
        },
        {
            "type": "technical_reasoning",
            "instruction": (
                "Present a scenario where the candidate must choose between methods, tools, models, "
                "or approaches. You want to see structured comparison, justified trade-offs, "
                "and awareness of constraints."
            ),
        },
        {
            "type": "situational",
            "instruction": (
                "Ask a case-style question involving ambiguous data, limited time, "
                "or conflicting business constraints. You want to see how the candidate "
                "prioritises and communicates under uncertainty."
            ),
        },
        {
            "type": "behavioral",
            "instruction": (
                "Ask how the candidate communicates technical work to non-technical stakeholders, "
                "collaborates across functions, or handles critical feedback. You want evidence "
                "of adaptability in communication style."
            ),
        },
    ],
}


EVALUATION_GUIDANCE = {
    "behavioral": (
        "WHAT THE INTERVIEWER REALLY WANTS: concrete evidence of the candidate's past behavior as a predictor of "
        "future performance—ownership, judgment, and self-awareness, not generic claims. "
        "For scoring: technical_accuracy = relevance and quality of the example, whether it demonstrates sound judgment. "
        "Reward clear STAR structure (Situation-Task-Action-Result) and genuine reflection on what was learned."
    ),
    "motivational": (
        "WHAT THE INTERVIEWER REALLY WANTS: a coherent narrative arc—how past experiences shaped the candidate's "
        "interest, why this specific role/company (not just any job), and what they hope to achieve. Authenticity "
        "and self-awareness matter more than rehearsed enthusiasm. "
        "For scoring: technical_accuracy = authenticity, genuine understanding of the role/company, "
        "and a believable connection between the candidate's trajectory and this opportunity."
    ),
    "situational": (
        "WHAT THE INTERVIEWER REALLY WANTS: to see how the candidate thinks under ambiguity—their prioritisation "
        "framework, risk awareness, and structured decision-making process. "
        "For scoring: technical_accuracy = practical reasoning quality, whether the candidate considers multiple "
        "angles (stakeholders, trade-offs, constraints), and proposes a clear action plan rather than vague platitudes."
    ),
    "resume_deep_dive": (
        "WHAT THE INTERVIEWER REALLY WANTS: to verify the candidate actually did the work and understands it deeply. "
        "Credibility comes from specific details, honest discussion of challenges, and reflection on what they would "
        "do differently. Vague or overly polished answers raise red flags. "
        "For scoring: technical_accuracy = specificity, credibility, appropriate level of detail, "
        "and honest acknowledgment of limitations or lessons learned."
    ),
    "technical": (
        "WHAT THE INTERVIEWER REALLY WANTS: evidence of applied problem-solving ability—can the candidate think "
        "through a real problem, choose appropriate tools/methods, and reason about trade-offs? "
        "For scoring: technical_accuracy = correctness, method/tool choice, trade-off awareness, "
        "and ability to explain reasoning clearly rather than just stating an answer."
    ),
    "technical_reasoning": (
        "WHAT THE INTERVIEWER REALLY WANTS: structured thinking and the ability to justify choices. The candidate "
        "should compare approaches on clear criteria (performance, cost, maintainability, etc.) and explain why "
        "one is preferable given the constraints. "
        "For scoring: technical_accuracy = structured comparison, justified choices, business or engineering "
        "trade-off awareness, and ability to articulate why one approach is better than another."
    ),
}


def available_interview_styles() -> list[str]:
    """Return supported mock interview styles."""
    return list(INTERVIEW_PLANS.keys())


@dataclass
class RubricScore:
    technical_accuracy: float
    completeness: float
    clarity: float
    overall: float
    feedback: str
    question: str
    answer: str
    question_type: str = "behavioral"
    interviewer_intent: str = ""


def generate_self_intro_draft(
    cv_text: str,
    jd_analysis: dict[str, Any],
    model: str = None,
) -> str:
    """Generate a tailored 60-second self-introduction draft."""
    role_summary = jd_analysis.get("summary", "the target role")
    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:6])
    model = model or get_model("gpt-4o-mini")

    prompt = (
        "Write a natural, first-person '60-second self-introduction' for a job interview.\n"
        "Base it ONLY on the CV text provided — do not invent facts.\n\n"
        f"Target role context: {role_summary}\n"
        f"Skills to highlight: {hard_skills}\n\n"
        f"CV text:\n{cv_text[:3000]}\n\n"
        "STRUCTURE (4-5 sentences):\n"
        "1. Open with who you are: current status (student/professional), institution or current role.\n"
        "2. Highlight your STRONGEST relevant project or achievement — be specific (name the project, "
        "the core skill used, and one concrete result).\n"
        "3. Plant a 'hook': briefly mention one interesting challenge or difficulty you encountered in "
        "that project WITHOUT explaining how you solved it. This is a deliberate conversation-starter "
        "designed to make the interviewer curious enough to ask a follow-up.\n"
        "4. Connect to this role: state a genuine, specific motivation for why this particular position "
        "excites you (not generic enthusiasm — tie it to your experience).\n\n"
        "STYLE: Use 'I' throughout. Confident but not arrogant. Vary sentence length. No bullet points. "
        "The hook should feel natural, not forced — e.g., 'During this project I had to navigate some "
        "unexpected data-quality issues, which ultimately shaped how I approach [skill]' rather than "
        "explicitly saying 'ask me about this'.\n"
        "Target: 80-120 words."
    )
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()


def _build_prep_system_prompt(
    jd_analysis: dict[str, Any],
    cv_text: str,
    self_intro: str,
    match_results: list[dict[str, Any]] | None = None,
) -> str:
    role_summary = jd_analysis.get("summary", "the target role")
    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:8])
    soft_skills = ", ".join(jd_analysis.get("soft_skills", [])[:5])
    interview_topics = "; ".join(jd_analysis.get("interview_topics", [])[:6])
    cv_snippet = cv_text[:2500] if cv_text else "(CV not provided)"
    intro_line = f"The candidate's self-introduction: {self_intro}\n" if self_intro else ""

    # Build skill-gap context from match results
    gap_context = ""
    if match_results:
        matched = [m.get("keyword", "") for m in match_results if m.get("score", 0) > 0.5]
        all_hard = jd_analysis.get("hard_skills", [])
        matched_lower = {s.lower() for s in matched}
        missing = [s for s in all_hard if s.lower() not in matched_lower]
        if missing:
            gap_context = (
                f"SKILL-GAP CONTEXT (from CV-JD matching):\n"
                f"- Strong matches: {', '.join(matched[:6])}\n"
                f"- Gaps (not found in CV): {', '.join(missing[:6])}\n"
                "Use these gaps as coaching opportunities: help the candidate frame them as growth "
                "motivation and learning goals rather than weaknesses.\n\n"
            )

    return (
        "You are a supportive, expert interview coach helping a candidate prepare for a job interview.\n\n"
        f"ROLE CONTEXT:\n"
        f"- Role summary: {role_summary}\n"
        f"- Key hard skills required: {hard_skills}\n"
        f"- Key soft skills required: {soft_skills}\n"
        f"- Likely interview topics: {interview_topics}\n\n"
        f"CANDIDATE BACKGROUND (from their CV):\n{cv_snippet}\n\n"
        f"{intro_line}"
        f"{gap_context}"
        "YOUR COACHING PHILOSOPHY:\n"
        "- Prepare FRAMEWORKS, not scripts. The candidate should walk in with structured thinking, "
        "not memorised word-for-word answers. Interviews are a conversation, not a recitation.\n"
        "- Every answer should be STRUCTURED. Teach the candidate to use the right framework for "
        "each question type:\n"
        "  • Behavioral questions → STAR (Situation-Task-Action-Result)\n"
        "  • Analytical/open questions → Conclusion first, then supporting points (pyramid principle)\n"
        "  • 'What would you do' questions → What is it → Why it matters → How I'd approach it\n"
        "  • If stuck with no clear framework → break into Internal/External or Subjective/Objective\n"
        "- For MOTIVATION questions, coach the candidate to build a narrative arc: "
        "past experience → impact it had on them → why it led them to this role → what they hope to achieve. "
        "Authenticity beats rehearsed enthusiasm.\n"
        "- When the candidate has SKILL GAPS, help them reframe gaps as growth motivation: "
        "'I haven't used X yet, but my experience with Y gave me a strong foundation, and learning X "
        "is exactly why this role excites me.'\n"
        "- Proactively help the candidate prepare 2-3 REVERSE QUESTIONS to ask the interviewer "
        "(e.g., team structure, day-to-day work, growth expectations, management style). "
        "Tailor these to the specific role.\n\n"
        "INTERVIEWER-AWARENESS TIPS (share when relevant):\n"
        "- Different interviewers prefer different answer structures. Strategic/consulting roles "
        "tend to prefer top-down logic (macro context → market → details). Operational/product/technical "
        "roles tend to prefer bottom-up logic (specific details → broader implications).\n"
        "- First-round interviewers (often the direct manager/mentor) typically focus on practical skills "
        "and work habits. Senior interviewers focus more on career thinking, communication, and collaboration.\n"
        "- Pay attention to how the interviewer phrases questions — their output style usually matches "
        "their preferred input style.\n\n"
        "COACHING STYLE:\n"
        "- When the candidate asks about a question or topic, first briefly explain what the "
        "interviewer is REALLY looking for (1-2 sentences).\n"
        "- Then suggest a concrete structured talking point drawn from their actual CV background. "
        "Be specific — reference real projects and experiences.\n"
        "- Encourage the candidate to embed 'hooks' in their answers — mentioning an interesting "
        "challenge or detail without fully expanding it, to invite follow-up questions on territory "
        "they are well-prepared for.\n"
        "- If the candidate shares a draft answer, give honest, constructive feedback: what works, "
        "what to add, what to cut, and how to make it more impactful.\n"
        "- Keep responses concise and practical. Use bullet points where helpful.\n"
        "- Never fabricate CV experiences. Only reference what is in the candidate's background.\n"
        "- Briefly remind the candidate about delivery when appropriate: speak with confidence and energy, "
        "project a positive first impression, and maintain conversational tone rather than reading a script."
    )


def prep_chat_summary(
    chat_history: list[dict[str, str]],
    jd_analysis: dict[str, Any],
    model: str = None,
) -> str:
    """Generate a bullet-point summary of what was discussed in the prep chat session."""
    model = model or get_model("gpt-4o-mini")
    role_summary = jd_analysis.get("summary", "the target role")

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in chat_history
    )
    prompt = (
        f"The following is a career coaching conversation for a candidate preparing for: {role_summary}\n\n"
        f"{conversation_text}\n\n"
        "Summarise this coaching session as a concise study reference. "
        "Use bullet points. Cover:\n"
        "- Which interview topics / questions were discussed\n"
        "- Key talking points or STAR examples the candidate should remember\n"
        "- Any improvement advice given\n"
        "Keep it under 200 words. Do not repeat the conversation verbatim."
    )
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def prep_chat_response(
    user_message: str,
    chat_history: list[dict[str, str]],
    jd_analysis: dict[str, Any],
    cv_text: str = "",
    self_intro: str = "",
    match_results: list[dict[str, Any]] | None = None,
    model: str = None,
) -> str:
    """Generate a coaching reply in the prep chat."""
    model = model or get_model("gpt-4o-mini")
    system_prompt = _build_prep_system_prompt(jd_analysis, cv_text, self_intro, match_results)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    resp = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


@dataclass
class InterviewSession:
    jd_analysis: dict[str, Any]
    match_results: list[dict[str, Any]]
    self_intro: str = ""
    interview_style: str = "1-to-1 interview"
    model: str = field(default_factory=lambda: get_model("gpt-4o-mini"))

    chat_history: list[dict] = field(default_factory=list)
    scores: list[RubricScore] = field(default_factory=list)
    _questions_asked: int = field(default=0, init=False)
    _max_questions: int = field(default=5, init=False)
    _current_question: str = field(default="", init=False)
    _current_question_type: str = field(default="", init=False)
    _question_plan: list[dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._question_plan = INTERVIEW_PLANS.get(
            self.interview_style,
            INTERVIEW_PLANS["1-to-1 interview"],
        )
        self._max_questions = len(self._question_plan)
        self.chat_history = [{"role": "system", "content": self._build_system_prompt()}]

    def next_question(self) -> str:
        """Ask the next planned interview question and return it."""
        if self._questions_asked >= self._max_questions:
            return ""

        slot = self._question_plan[self._questions_asked]
        question_type = slot["type"]
        type_label = QUESTION_TYPE_LABELS.get(question_type, question_type.title())
        previous_questions = " | ".join(score.question for score in self.scores[-3:]) or "(none)"

        if self._questions_asked == 0:
            instruction = (
                "Begin with a warm one-sentence greeting, then immediately ask exactly one "
                f"{type_label.lower()} question. {slot['instruction']} "
                f"Previous questions: {previous_questions}. "
                "Keep the whole response under 2 sentences."
            )
        else:
            instruction = (
                f"Ask exactly one {type_label.lower()} interview question. "
                f"{slot['instruction']} Previous questions: {previous_questions}. "
                "Do not repeat earlier topics. Keep it under 24 words. Output only the question."
            )

        self.chat_history.append({"role": "user", "content": instruction})
        reply = self._call_llm(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": reply})
        self._current_question = reply
        self._current_question_type = question_type
        self._questions_asked += 1
        return reply

    def answer(self, user_reply: str) -> RubricScore:
        """Submit the candidate's answer and return the rubric score."""
        if not self._current_question:
            raise RuntimeError("Call next_question() before answer()")

        self.chat_history.append({"role": "user", "content": user_reply})
        score = self._evaluate(self._current_question, user_reply, self._current_question_type)

        ack_instruction = (
            "Acknowledge the candidate's answer briefly in 1 sentence without giving explicit scores. "
            "If this is not the last question, optionally add a short transition."
        )
        self.chat_history.append({"role": "user", "content": ack_instruction})
        ack = self._call_llm(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": ack})

        self.scores.append(score)
        return score

    def final_report(self) -> dict[str, Any]:
        """Return a summary report after all questions have been answered."""
        if not self.scores:
            return {"error": "No answers recorded yet"}

        avg_technical = sum(score.technical_accuracy for score in self.scores) / len(self.scores)
        avg_completeness = sum(score.completeness for score in self.scores) / len(self.scores)
        avg_clarity = sum(score.clarity for score in self.scores) / len(self.scores)
        avg_overall = sum(score.overall for score in self.scores) / len(self.scores)

        per_type: dict[str, list[RubricScore]] = {}
        for score in self.scores:
            per_type.setdefault(score.question_type, []).append(score)

        per_type_summary = []
        for question_type, items in per_type.items():
            per_type_summary.append({
                "question_type": question_type,
                "label": QUESTION_TYPE_LABELS.get(question_type, question_type.title()),
                "rounds": len(items),
                "avg_technical_accuracy": round(sum(item.technical_accuracy for item in items) / len(items), 2),
                "avg_completeness": round(sum(item.completeness for item in items) / len(items), 2),
                "avg_clarity": round(sum(item.clarity for item in items) / len(items), 2),
                "avg_overall": round(sum(item.overall for item in items) / len(items), 2),
            })

        closing_prompt = (
            "The interview is now complete. Provide a 4-6 sentence holistic assessment that includes:\n"
            "1. Overall impression of the candidate's performance across the interview.\n"
            "2. Their strongest area and a specific example from their answers.\n"
            "3. One concrete area for improvement with a specific suggestion.\n"
            "4. A reflection tip: for each question type covered, briefly note what the interviewer "
            "was really testing — help the candidate understand the pattern so they can improve "
            "for future interviews.\n"
            "5. One piece of strategic advice for their next interview (e.g., structure, hooks, framing)."
        )
        self.chat_history.append({"role": "user", "content": closing_prompt})
        closing = self._call_llm(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": closing})

        return {
            "interview_style": self.interview_style,
            "rounds": len(self.scores),
            "question_types_covered": [score.question_type for score in self.scores],
            "avg_technical_accuracy": round(avg_technical, 2),
            "avg_completeness": round(avg_completeness, 2),
            "avg_clarity": round(avg_clarity, 2),
            "avg_overall": round(avg_overall, 2),
            "per_type_summary": per_type_summary,
            "per_round": [
                {
                    "round": index + 1,
                    "question": score.question,
                    "question_type": score.question_type,
                    "technical_accuracy": score.technical_accuracy,
                    "completeness": score.completeness,
                    "clarity": score.clarity,
                    "overall": score.overall,
                    "feedback": score.feedback,
                    "interviewer_intent": score.interviewer_intent,
                }
                for index, score in enumerate(self.scores)
            ],
            "closing_assessment": closing,
        }

    def _build_system_prompt(self) -> str:
        hard_skills = ", ".join(self.jd_analysis.get("hard_skills", [])[:6]) or "Not specified"
        soft_skills = ", ".join(self.jd_analysis.get("soft_skills", [])[:5]) or "Not specified"
        role_summary = self.jd_analysis.get("summary", "a role")
        interview_topics = "; ".join(self.jd_analysis.get("interview_topics", [])[:6]) or "Not specified"
        cv_highlights = " | ".join(
            match.get("suggested_rewrite", match.get("original", ""))[:120]
            for match in (self.match_results or [])[:3]
        ) or "No CV matching context available"
        intro_context = (
            f"The candidate introduced themselves as: {self.self_intro}\n"
            if self.self_intro else ""
        )
        plan_summary = ", ".join(
            QUESTION_TYPE_LABELS.get(slot["type"], slot["type"].title())
            for slot in self._question_plan
        )

        return (
            f"You are a friendly but professional interviewer running a {self.interview_style}.\n"
            f"Role summary: {role_summary}.\n"
            f"Key hard skills: {hard_skills}.\n"
            f"Key soft skills: {soft_skills}.\n"
            f"Likely interview topics: {interview_topics}.\n"
            f"Candidate background highlights: {cv_highlights}.\n"
            f"{intro_context}"
            f"Planned question categories for this interview: {plan_summary}.\n"
            f"Ask exactly {self._max_questions} questions total, one per turn.\n"
            "Keep each question concise and role-relevant.\n"
            "Never ask technical trivia, tool syntax recall, or duplicate earlier topics.\n"
            "Follow the user's explicit instruction about which question category to ask next."
        )

    def _call_llm(self, messages: list[dict]) -> str:
        resp = get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()

    def _evaluate(self, question: str, answer: str, question_type: str) -> RubricScore:
        """Call the LLM as an evaluator and parse rubric JSON."""
        guidance = EVALUATION_GUIDANCE.get(question_type, EVALUATION_GUIDANCE["behavioral"])
        eval_prompt = (
            "You are an objective interview evaluator. Score the following answer on three dimensions "
            "(each 1-5 where 1=poor and 5=excellent), then compute an overall weighted average "
            "(technical_accuracy * 0.4 + completeness * 0.35 + clarity * 0.25).\n\n"
            f"Question type: {question_type}\n"
            f"Evaluation guidance: {guidance}\n\n"
            f"Interview question: \"{question}\"\n\n"
            f"Candidate answer: \"{answer}\"\n\n"
            "Return ONLY a JSON object with keys: "
            "technical_accuracy (float), completeness (float), clarity (float), "
            "overall (float), feedback (str, max 2 sentences of actionable advice), "
            "interviewer_intent (str, 1 sentence explaining what the interviewer was really "
            "trying to assess with this specific question — help the candidate understand "
            "the hidden purpose behind the question)."
        )

        resp = get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "technical_accuracy": 3.0,
                "completeness": 3.0,
                "clarity": 3.0,
                "overall": 3.0,
                "feedback": "Could not parse rubric.",
                "interviewer_intent": "",
            }

        return RubricScore(
            technical_accuracy=float(data.get("technical_accuracy", 3)),
            completeness=float(data.get("completeness", 3)),
            clarity=float(data.get("clarity", 3)),
            overall=float(data.get("overall", 3)),
            feedback=data.get("feedback", ""),
            question=question,
            answer=answer,
            question_type=question_type,
            interviewer_intent=data.get("interviewer_intent", ""),
        )
