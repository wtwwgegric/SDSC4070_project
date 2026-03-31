"""Interview Simulator — multi-turn interviewer with per-answer rubric scoring.

Usage
-----
session = InterviewSession(jd_analysis, match_results)
question = session.next_question()          # get the first question
score    = session.answer(user_reply)       # submit answer → get RubricScore
question = session.next_question()          # get follow-up or next question
...
report   = session.final_report()           # summary after the interview

RubricScore fields (each 1–5):
  technical_accuracy  — correctness / relevance to the JD requirement
  completeness        — depth of the answer (covers the key points)
  clarity             — logical flow and communication quality
  overall             — weighted average
  feedback            — short qualitative comment
"""
import json
from dataclasses import dataclass, field
from typing import Any

from career_copilot.config import get_client, get_model


@dataclass
class RubricScore:
    technical_accuracy: float
    completeness: float
    clarity: float
    overall: float
    feedback: str
    question: str
    answer: str


def generate_self_intro_draft(
    cv_text: str,
    jd_analysis: dict[str, Any],
    model: str = None,
) -> str:
    """Generate a tailored 60-second self-introduction draft.

    The draft is based on the candidate's own CV text and the target JD so they
    can practise or refine it before the interview starts.
    """
    role_summary = jd_analysis.get("summary", "the target role")
    hard_skills = ", ".join(jd_analysis.get("hard_skills", [])[:6])
    model = model or get_model("gpt-4o-mini")

    prompt = (
        "Write a natural, first-person '60-second self-introduction' for a job interview.\n"
        "Base it ONLY on the CV text provided — do not invent facts.\n"
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


@dataclass
class InterviewSession:
    jd_analysis: dict[str, Any]
    match_results: list[dict[str, Any]]
    self_intro: str = ""                   # candidate's confirmed self-introduction
    model: str = field(default_factory=lambda: get_model("gpt-4o-mini"))

    # internal state
    chat_history: list[dict] = field(default_factory=list)
    scores: list[RubricScore] = field(default_factory=list)
    _questions_asked: int = field(default=0, init=False)
    _max_questions: int = field(default=5, init=False)
    _current_question: str = field(default="", init=False)

    def __post_init__(self):
        self.chat_history = [{"role": "system", "content": self._build_system_prompt()}]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_question(self) -> str:
        """Ask the interviewer for the next question. Returns the question text."""
        if self._questions_asked >= self._max_questions:
            return ""

        instruction = (
            "Ask your next interview question. "
            "Follow the rules in your system prompt. "
            "Output ONLY the question — no numbering, no preamble."
        )
        if self._questions_asked == 0:
            instruction = (
                "Begin the interview with a warm one-sentence greeting, "
                "then immediately ask your first behavioural question. "
                "Keep the total response under 2 sentences."
            )

        self.chat_history.append({"role": "user", "content": instruction})
        reply = self._call_llm(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": reply})
        self._current_question = reply
        self._questions_asked += 1
        return reply

    def answer(self, user_reply: str) -> RubricScore:
        """Submit the candidate's answer. Returns a RubricScore with feedback."""
        if not self._current_question:
            raise RuntimeError("Call next_question() before answer()")

        self.chat_history.append({"role": "user", "content": user_reply})

        # Get rubric score from a dedicated evaluation call (separate from conversation)
        score = self._evaluate(self._current_question, user_reply)

        # Let the interviewer acknowledge and follow up
        ack_instruction = (
            "Acknowledge the candidate's answer briefly (1 sentence, no evaluation). "
            "If this is not the last question, optionally add a short follow-up thought."
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

        avg_technical = sum(s.technical_accuracy for s in self.scores) / len(self.scores)
        avg_completeness = sum(s.completeness for s in self.scores) / len(self.scores)
        avg_clarity = sum(s.clarity for s in self.scores) / len(self.scores)
        avg_overall = sum(s.overall for s in self.scores) / len(self.scores)

        # Ask LLM for a holistic closing remark
        closing_prompt = (
            "The interview is now complete. Provide a brief (3-4 sentence) holistic assessment "
            "of the candidate's performance: strengths observed, one area to improve, and "
            "an encouraging closing remark."
        )
        self.chat_history.append({"role": "user", "content": closing_prompt})
        closing = self._call_llm(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": closing})

        return {
            "rounds": len(self.scores),
            "avg_technical_accuracy": round(avg_technical, 2),
            "avg_completeness": round(avg_completeness, 2),
            "avg_clarity": round(avg_clarity, 2),
            "avg_overall": round(avg_overall, 2),
            "per_round": [
                {
                    "round": i + 1,
                    "question": s.question,
                    "technical_accuracy": s.technical_accuracy,
                    "completeness": s.completeness,
                    "clarity": s.clarity,
                    "overall": s.overall,
                    "feedback": s.feedback,
                }
                for i, s in enumerate(self.scores)
            ],
            "closing_assessment": closing,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        hard_skills = ", ".join(self.jd_analysis.get("hard_skills", [])[:6])
        role_summary = self.jd_analysis.get("summary", "a role")
        cv_highlights = " | ".join(
            m.get("suggested_rewrite", m.get("original", ""))[:100]
            for m in (self.match_results or [])[:3]
        )
        intro_context = (
            f"The candidate introduced themselves as: {self.self_intro}\n"
            if self.self_intro else ""
        )
        return (
            f"You are a friendly but professional interviewer for: {role_summary}.\n"
            f"The role requires these skills: {hard_skills}.\n"
            f"Candidate background highlights: {cv_highlights}.\n"
            f"{intro_context}"
            "STRICT RULES for asking questions:\n"
            f"- Ask exactly {self._max_questions} questions total, one per turn.\n"
            "- Keep each question SHORT — ideally under 20 words.\n"
            "- Question mix: at least 3 behavioural (STAR: 'Tell me about a time...'),"
            " 1 motivational ('Why this role / company?'), 1 situational ('How would you handle...').\n"
            "- Do NOT ask technical trivia or tool-syntax questions.\n"
            "- Probe mindset, soft skills, and past impact.\n"
            "- Never repeat a topic already covered."
        )

    def _call_llm(self, messages: list[dict]) -> str:
        resp = get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()

    def _evaluate(self, question: str, answer: str) -> RubricScore:
        """Call LLM as an evaluator (separate from conversation) and parse rubric JSON."""
        eval_prompt = (
            "You are an objective interview evaluator. Score the following answer on three dimensions "
            "(each 1–5 where 1=poor and 5=excellent), then compute an overall weighted average "
            "(technical_accuracy × 0.4 + completeness × 0.35 + clarity × 0.25).\n\n"
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
            max_tokens=200,
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
        )
