# Career Co-pilot

An AI-powered job search assistant that helps you analyze job descriptions, optimize your CV, generate cover letters, and prepare for interviews — all grounded in your own CV content.

---

## Features

### 📋 Tab 1 — JD Analyzer
Paste any job description and get structured insights:
- **Hard skills** extracted (programming languages, tools, frameworks)
- **Soft skills** identified
- **Jargon decoder** — corporate buzzwords explained in plain English with interview tips
- **Interview topics** — 5–8 likely themes the interviewer will probe
- **Role summary** — one-paragraph plain-English overview of the position

### 🎯 Tab 2 — CV Matcher
Upload your CV (PDF) and match it against the analyzed JD:
- **Value Refiner** — select any CV chunk and rewrite it as a polished, impact-focused bullet point
- **RAG matching** — indexes your CV into a ChromaDB vector store and retrieves passages most relevant to each JD requirement, with LLM-generated rewrite suggestions
- **Keyword hit rate metrics** — before/after comparison showing how much the rewrites improve JD keyword coverage
- **Chunk preview** — inspect all CV sections to verify the paragraph-aware chunker is working correctly

### ✉️ Tab 3 — Cover Letter Generator
Generates a professional cover letter grounded entirely in your CV:
- Follows a structured format: salutation → opening (role + status) → two body paragraphs mapping CV evidence to JD requirements → closing call-to-action → sign-off
- Every claim is traceable to your CV — no fabricated facts
- Downloadable as a `.txt` file

### 🎤 Tab 4 — Interview Simulator
Two modes selectable at the top of the tab:

**🏋️ Prep Chat (default)**
Free-form coaching conversation with an AI career coach who knows your JD and CV:
- Ask about likely questions: *"What might they ask about my Python experience?"*
- Get STAR-structured talking points drawn from your actual CV background
- Share a draft answer for honest feedback: *"Is this a good answer: …"*
- Ask for strategy: *"Which topics should I prioritise?"*
- Full conversation history displayed; "🗑️ Clear chat" to start fresh

**📝 Mock Test**
Structured 5-question interview simulation:
- Starts with a self-introduction phase — generate a draft from your CV or write your own
- 5 questions mixing behavioural (STAR), motivational, and situational types
- Per-answer rubric scoring: Technical Accuracy, Completeness, Clarity (each 1–5)
- Feedback shown after each answer; advance at your own pace
- Final report with score charts (line chart per round), averages, and a closing assessment
- Self-introduction stays visible as a reference panel throughout the interview

### 🔍 Sidebar — Company Culture Lookup
Enter a company name to fetch cultural hints from web sources via the Serper.dev API (requires a valid API key at [serper.dev](https://serper.dev)).

---

## Project Structure

```
app.py                          # Streamlit app — 4 tabs
requirements.txt
.env                            # API keys (git-ignored)
.env.example                    # Template for .env
career_copilot/
    config.py                   # Central API client factory (get_client / get_model)
    pdf_loader.py               # PDF extraction (pypdf + pdfminer.six) + paragraph-aware chunker
    rag.py                      # ChromaDB vector store — index CV chunks + query
    jd_analyzer.py              # JD → structured JSON (hard_skills, soft_skills, jargon, topics)
    cv_matcher.py               # RAG match CV against JD keywords + LLM rewrite suggestions
    value_refiner.py            # Rewrite a CV description into a polished bullet point
    cover_letter.py             # Grounded cover letter generator
    interview_simulator.py      # Prep chat coach + mock test session with rubric scoring
    serper.py                   # Serper.dev wrapper for company culture search
    agent.py                    # LangGraph pipeline wiring all nodes
    eval_metrics.py             # Quantitative metrics (keyword hit rate, hallucination check)
examples/
    rag_demo.py                 # Standalone RAG demo (index PDF + query from CLI)
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
# .env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # remove for standard OpenAI
OPENAI_MODEL=qwen-plus            # or gpt-4o-mini for OpenAI
OPENAI_EMBED_MODEL=text-embedding-v3   # or text-embedding-3-small for OpenAI
SERPER_API_KEY=your_serper_key    # optional — needed for company culture lookup
```

> **Using Qwen / DashScope:** set `OPENAI_BASE_URL` to the DashScope compatible endpoint. The rest of the code uses the standard OpenAI SDK and requires no other changes.
>
> **Using standard OpenAI:** leave `OPENAI_BASE_URL` unset (or remove it) and use `gpt-4o-mini` / `text-embedding-3-small` as model names.

---

## Running the App

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Recommended workflow:**
1. Upload your CV PDF in the sidebar and enter your name + target company
2. **Tab 1** — Paste the JD and click "Analyze JD"
3. **Tab 2** — Click "Run CV Matching" to see keyword gaps and rewrite suggestions
4. **Tab 3** — Click "Generate Cover Letter"
5. **Tab 4** — Use "Prep Chat" to practice with the AI coach, then "Mock Test" for a full simulation

---

## Dependencies (key packages)

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | — | Web UI |
| `openai` | 2.30.0 | LLM + embeddings API (OpenAI-compatible) |
| `chromadb` | 1.5.5 | Vector store for CV RAG |
| `langgraph` | 1.1.3 | Agent pipeline orchestration |
| `langchain` | 1.2.13 | LangChain utilities |
| `pypdf` | — | PDF text extraction (primary) |
| `pdfminer.six` | — | PDF text extraction (fallback) |
| `python-dotenv` | 1.2.2 | `.env` file loading |
| `pandas` | — | Score charts and data display |
| `requests` | — | Serper.dev HTTP calls |

---

## Notes

- **Scanned / image PDFs** will extract 0 characters. Use a text-based PDF or copy-paste your CV text directly into the Value Refiner.
- **Serper API key**: the free tier at [serper.dev](https://serper.dev) provides a limited number of searches. The key must be the short alphanumeric string from your dashboard, not a hex hash.
- **CV chunking**: the paragraph-aware chunker splits on blank lines (natural CV section boundaries) rather than fixed character windows, so each chunk corresponds to a meaningful section of your CV.

