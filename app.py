import os
import streamlit as st
from career_copilot.pdf_loader import load_pdf_from_bytes, chunk_text
from career_copilot.value_refiner import refine_value
from career_copilot.serper import fetch_company_culture


st.set_page_config(page_title="Career Co-pilot — Demo", layout="wide")

st.title("Career Co-pilot — Value Refiner Demo")

st.markdown("Upload a CV (PDF) or paste a short 'dirty work' line to see it transformed.")

uploaded = st.file_uploader("Upload PDF CV", type=["pdf"])

raw_text = ""
if uploaded is not None:
    bytes_data = uploaded.read()
    with st.spinner("Extracting text..."):
        raw_text = load_pdf_from_bytes(bytes_data)
    st.success("PDF text extracted — showing first 4000 characters")
    st.text_area("Extracted text", raw_text[:4000], height=250)

manual = st.text_area("Or paste a single Dirty-Work line to refine (e.g. '每天手動修復數據錯誤')")

example = "設計並維護資料處理腳本，每日手動修復資料不一致問題。"
if st.button("Use example"):
    manual = example

input_text = manual.strip()
if not input_text and raw_text:
    # if no manual input, auto-chunk and allow selecting a chunk
    chunks = chunk_text(raw_text, chunk_size=1200, overlap=200)
    sel = st.selectbox("Select a chunk to refine", options=list(range(min(10, len(chunks)))), format_func=lambda i: f"Chunk {i+1}: {chunks[i][:80].strip()}...")
    if sel is not None:
        input_text = chunks[sel]

if input_text:
    st.subheader("Input to refine")
    st.write(input_text[:1000])

    if st.button("Refine Value" ):
        with st.spinner("Calling the ValueRefiner..."):
            try:
                out = refine_value(input_text)
            except Exception as e:
                st.error(f"Error calling LLM: {e}")
            else:
                st.subheader("Refined output")
                st.write(out)

st.sidebar.header("Notes")
st.sidebar.write("Make sure to set environment variable `OPENAI_API_KEY` before running:")
st.sidebar.code("export OPENAI_API_KEY=your_key_here")
st.sidebar.write("")
st.sidebar.write("Optional: set `SERPER_API_KEY` to enable company-culture lookups.")
company = st.sidebar.text_input("Company name for culture lookup")
if st.sidebar.button("Fetch company culture") and company:
    try:
        with st.spinner(f"Fetching culture hints for {company}..."):
            hits = fetch_company_culture(company, num_results=5)
    except Exception as e:
        st.sidebar.error(f"Error fetching from Serper.dev: {e}")
        hits = []
    if hits:
        st.sidebar.markdown("**Top culture hints**")
        for h in hits:
            title = h.get("title")
            snippet = h.get("snippet")
            link = h.get("link")
            if title:
                st.sidebar.write(f"**{title}**")
            if snippet:
                st.sidebar.write(snippet)
            if link:
                st.sidebar.write(link)
            st.sidebar.write("---")
