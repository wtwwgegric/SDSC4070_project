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
    """Simple sliding window chunker by characters."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks
