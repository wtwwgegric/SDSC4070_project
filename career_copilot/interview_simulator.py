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
                "problem-solving, or resilience."
            ),
        },
        {
            "type": "motivational",
            "instruction": (
                "Ask why the candidate wants this role, this company, or how the role fits "
                "their goals."
            ),
        },
        {
            "type": "situational",
            "instruction": (
                "Ask how the candidate would handle ambiguity, conflicting priorities, "
                "or stakeholder pressure in this role."
            ),
        },
        {
            "type": "resume_deep_dive",
            "instruction": (
                "Ask a specific follow-up about one relevant project, internship, or achievement "
                "from the candidate's background."
            ),
        },
        {
            "type": "behavioral",
            "instruction": (
                "Ask another behavioral question, but focus on collaboration, communication, "
                "or learning from failure instead of repeating earlier themes."
            ),
        },
    ],
    "Technical interview": [
        {
            "type": "technical",
            "instruction": (
                "Ask a practical technical question tied to the role's core hard skills. "
                "Focus on applied reasoning, not trivia or syntax recall."
            ),
        },
        {
            "type": "resume_deep_dive",
            "instruction": (
                "Ask the candidate to explain a technically relevant project or task from their "
                "background in more depth."
            ),
        },
        {
            "type": "technical_reasoning",
            "instruction": (
                "Ask how the candidate would choose between methods, tools, models, or approaches "
                "for a realistic role-related task."
            ),
        },
        {
            "type": "situational",
            "instruction": (
                "Ask a case-style or trade-off question involving ambiguous data, limited time, "
                "or conflicting business constraints."
            ),
        },
        {
            "type": "behavioral",
            "instruction": (
                "Ask how the candidate communicates technical work, collaborates with teammates, "
                "or handles feedback in cross-functional settings."
            ),
        },
    ],
}


EVALUATION_GUIDANCE = {
    "behavioral": (
        "For behavioral questions, technical_accuracy means relevance of the example, quality of evidence, "
        "and whether the answer demonstrates sound judgment. Reward clear STAR structure and reflection."
    ),
    "motivational": (
        "For motivational questions, technical_accuracy means authenticity, understanding of the role/company, "
        "and a believable alignment between the candidate and the opportunity."
    ),
    "situational": (
        "For situational questions, technical_accuracy means practical reasoning, prioritisation, risk awareness, "
        "and decision quality rather than factual recall."
    ),
    "resume_deep_dive": (
        "For resume deep-dive questions, technical_accuracy means whether the candidate explains their own work "
        "credibly, specifically, and at an appropriate level of detail."
    ),
    "technical": (
        "For technical questions, technical_accuracy means correctness, method/tool choice, and awareness of trade-offs."
    ),
    "technical_reasoning": (
        "For technical reasoning questions, technical_accuracy means structured thinking, justified choices, business or "
        "engineering trade-offs, and ability to explain why one approach is better than another."
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
        "Base it ONLY on the CV text provided - do not invent facts.\n"
        f"Target role context: {role_summary}\n"
        f"Skills to highlight: {hard_skills}\n\n"
        f"CV text:\n{cv_text[:3000]}\n\n"
        "Format: 3-4 short sentences. Start with the candidate's current status, "
        "briefly mention 1-2 relevant past achievements, then state motivation for this role. "
        "Use 'I' throughout. No bullet points."
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
) -> str:
    role_summary = jd_analysis.get("summary", "the target role")
    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:8])
    soft_skills = ", ".join(jd_analysis.get("soft_skills", [])[:5])
    interview_topics = "; ".join(jd_analysis.get("interview_topics", [])[:6])
    cv_snippet = cv_text[:2500] if cv_text else "(CV not provided)"
    intro_line = f"The candidate's self-introduction: {self_intro}\n" if self_intro else ""

    return (
        "You are a supportive and expert interview coach helping a candidate prepare for a job interview.\n\n"
        f"ROLE CONTEXT:\n"
        f"- Role summary: {role_summary}\n"
        f"- Key hard skills required: {hard_skills}\n"
        f"- Key soft skills required: {soft_skills}\n"
        f"- Likely interview topics: {interview_topics}\n\n"
        f"CANDIDATE BACKGROUND (from their CV):\n{cv_snippet}\n\n"
        f"{intro_line}"
        "YOUR COACHING STYLE:\n"
        "- When the candidate asks about a question or topic, first briefly explain what the "
        "interviewer is really looking for (1-2 sentences).\n"
        "- Then suggest a concrete STAR-structured talking point they could use, drawn from "
        "their actual CV background above. Be specific - reference real projects/experiences.\n"
        "- If the candidate shares a draft answer, give honest, constructive feedback: what works, "
        "what to add, what to cut, and how to make it more impactful.\n"
        "- Keep responses concise and practical. Use bullet points where helpful.\n"
        "- Never fabricate CV experiences. Only reference what is in the candidate's background.\n"
        "- You may proactively suggest which interview topics the candidate should prioritise "
        "based on the JD requirements."
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
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def prep_chat_response(
    user_message: str,
    chat_history: list[dict[str, str]],
    jd_analysis: dict[str, Any],
    cv_text: str = "",
    self_intro: str = "",
    model: str = None,
) -> str:
    """Generate a coaching reply in the prep chat."""
    model = model or get_model("gpt-4o-mini")
    system_prompt = _build_prep_system_prompt(jd_analysis, cv_text, self_intro)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    resp = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=600,
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
            "The interview is now complete. Provide a brief 3-4 sentence holistic assessment of the "
            "candidate's performance, explicitly considering the interview style, question variety, "
            "strongest answer area, and one improvement area."
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
            max_tokens=400,
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
            "overall (float), feedback (str, max 2 sentences)."
        )

        resp = get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=220,
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
        )
