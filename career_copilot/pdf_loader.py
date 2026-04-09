import logging
import re
import warnings
from io import BytesIO
from typing import List

# Suppress harmless pdfminer/pdfplumber font-parsing warnings
# (e.g. "Could not get FontBBox from font descriptor")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")


def _fix_spaced_chars(text: str) -> str:
    """Fix PDFs where every glyph is stored with explicit spacing metadata,
    producing output like 'P r o j e c t' instead of 'Project'.

    Detection: if >45% of whitespace-split tokens are single characters,
    the document is almost certainly using this encoding. We then treat
    single spaces as intra-word separators and double-spaces as word boundaries.
    """
    if not text:
        return text
    tokens = re.split(r'\s+', text)
    non_empty = [t for t in tokens if t]
    if len(non_empty) < 20:
        return text  # too short to make a reliable judgment
    single_ratio = sum(1 for t in non_empty if len(t) == 1) / len(non_empty)
    if single_ratio < 0.45:
        return text  # normal text — leave untouched

    fixed_lines = []
    for line in text.split('\n'):
        if not line.strip():
            fixed_lines.append('')
            continue
        # Double (or more) spaces mark word boundaries;
        # single spaces are inter-character gaps within a word.
        parts = re.split(r'  +', line)
        fixed_parts = []
        for part in parts:
            collapsed = re.sub(r'(?<=\S) (?=\S)', '', part).strip()
            if collapsed:
                fixed_parts.append(collapsed)
        fixed_lines.append(' '.join(fixed_parts))
    return '\n'.join(fixed_lines)


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Best-effort column-aware extractor using pdfplumber.

    pdfplumber groups text objects by spatial position, so it handles
    two-column CV layouts (sidebar + main area) much better than pypdf
    or raw pdfminer.
    """
    import pdfplumber
    pages = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(
                x_tolerance=3,
                y_tolerance=3,
                layout=True,          # preserve spatial layout
            )
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_with_pypdf(pdf_bytes: bytes) -> str:
    """Fallback extractor using pypdf."""
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
    """Last-resort extractor using pdfminer.six."""
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    import io
    output = io.StringIO()
    extract_text_to_fp(
        BytesIO(pdf_bytes),
        output,
        laparams=LAParams(boxes_flow=None),   # None = respect PDF element order
        output_type="text",
        codec="utf-8",
    )
    return output.getvalue()


def load_pdf_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes.

    Extraction order (first non-empty result wins):
      1. pdfplumber  — best for multi-column CVs
      2. pypdf       — fast, works well for single-column
      3. pdfminer    — last resort, different layout heuristic

    Applies spaced-character post-processing for PDFs with glyph-level spacing.
    Returns an empty string (not an exception) for image-only/scanned PDFs.
    """
    # Try pdfplumber first — handles two-column CVs
    text = ""
    try:
        text = _extract_with_pdfplumber(pdf_bytes).strip()
    except Exception:
        pass

    # Fallback: pypdf
    if not text:
        try:
            text = _extract_with_pypdf(pdf_bytes).strip()
        except Exception:
            pass

    # Last resort: pdfminer
    if not text:
        try:
            text = _extract_with_pdfminer(pdf_bytes).strip()
        except Exception:
            pass

    return _fix_spaced_chars(text)


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
