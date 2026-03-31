import re
from io import BytesIO
from typing import List


def _extract_with_pypdf(pdf_bytes: bytes) -> str:
    """Primary extractor using pypdf (maintained successor to PyPDF2)."""
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def _extract_with_pdfminer(pdf_bytes: bytes) -> str:
    """Fallback extractor using pdfminer.six — handles more encoding edge cases."""
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    import io
    output = io.StringIO()
    extract_text_to_fp(
        BytesIO(pdf_bytes),
        output,
        laparams=LAParams(),
        output_type="text",
        codec="utf-8",
    )
    return output.getvalue()


def load_pdf_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes.

    Tries pypdf first; falls back to pdfminer.six if the result is empty.
    Returns an empty string (not an exception) for image-only/scanned PDFs.
    """
    text = ""
    try:
        text = _extract_with_pypdf(pdf_bytes).strip()
    except Exception:
        pass

    if not text:
        try:
            text = _extract_with_pdfminer(pdf_bytes).strip()
        except Exception:
            pass

    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Paragraph-aware chunker for CV text.

    Strategy:
    1. Split by blank lines to get natural paragraphs / CV sections.
    2. Accumulate paragraphs into a chunk until size is reached.
    3. Paragraphs that are themselves too long are split by sentence.
    This produces chunks that align with job entries, bullet points, and
    education blocks rather than cutting inside a sentence.
    """
    if not text:
        return []

    # Step 1: split into paragraphs on one or more blank lines
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    def _split_long_para(para: str) -> List[str]:
        """Split a paragraph that exceeds chunk_size by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', para)
        parts: List[str] = []
        current = ""
        for s in sentences:
            candidate = (current + " " + s).strip() if current else s
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = s
        if current:
            parts.append(current)
        return parts or [para]

    chunks: List[str] = []
    current = ""

    for para in raw_paras:
        # If the paragraph fits alongside what we already have, merge
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Save current chunk
            if current:
                chunks.append(current)
            # Handle oversized paragraph
            if len(para) > chunk_size:
                sub_parts = _split_long_para(para)
                chunks.extend(sub_parts[:-1])
                current = sub_parts[-1]
            else:
                current = para

    if current:
        chunks.append(current)

    # Fallback: if nothing produced (e.g. no blank lines at all), use sliding window
    if not chunks:
        step = max(1, chunk_size - overlap)
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]

    return chunks
