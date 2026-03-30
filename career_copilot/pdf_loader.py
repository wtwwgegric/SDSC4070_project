from io import BytesIO
from typing import List
import PyPDF2


def load_pdf_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # best-effort extraction
            pages.append("")
    return "\n\n".join(pages)


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
