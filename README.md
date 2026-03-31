Getting started (Career Co-pilot scaffold)

- Install dependencies:

```bash
pip install -r ../requirements.txt
```

- Run the demo app:

```bash
export OPENAI_API_KEY=your_key_here

source /Users/wantwa/SDSC4070/SDSC4070_project/.venv/bin/activate
streamlit run app.py
```

This scaffold includes:

- `career_copilot/pdf_loader.py`: PDF extraction + simple chunker
- `career_copilot/value_refiner.py`: small OpenAI wrapper to turn "dirty work" into value-focused bullets
- `app.py`: minimal Streamlit demo to upload a CV or paste a task and refine it

Next steps you may ask me to implement:

- RAG indexing with Chroma or FAISS for CV search
- Serper.dev integration to fetch company culture hints
- A more structured prompt template and unit tests

RAG quickstart

- Install/update requirements (Chroma and LangChain are required):

```bash
pip install -r ../requirements.txt
pip install chromadb langchain
```

- Example: index a CV PDF and query:

```bash
python ../examples/rag_demo.py path/to/your_cv.pdf
python ../examples/rag_demo.py   # then enter a query
```

By default the demo persists Chroma to `./chromadb`.

Serper.dev integration

- To enable company-culture lookups, set `SERPER_API_KEY` in your environment. Optionally set `SERPER_URL` to override the endpoint.

```bash
export SERPER_API_KEY=your_serper_key
```

- In the Streamlit demo sidebar enter a company name and click "Fetch company culture" to see top snippets from public web sources.
