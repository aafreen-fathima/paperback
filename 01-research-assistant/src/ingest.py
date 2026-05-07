"""PDF ingestion: parse, chunk by section (with fallback), embed, store in Chroma."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import pdfplumber
import tiktoken
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
FALLBACK_CHUNK_TOKENS = 500
FALLBACK_OVERLAP = 50
# Hard cap on any chunk going to the embedding API. OpenAI's limit is 8192;
# we stay well under for retrieval quality and to avoid edge cases.
MAX_CHUNK_TOKENS = 1500

# Headings we treat as section boundaries. Order doesn't matter — we match anywhere.
SECTION_HEADERS = {
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach",
    "experiments", "results", "evaluation",
    "discussion", "limitations", "conclusion", "conclusions",
    "references", "appendix",
}

# Lines on page 1 that look like attribution / copyright notices, not titles.
ATTRIBUTION_NOISE = (
    "permission", "provided proper attribution", "google hereby",
    "creative commons", "all rights reserved", "copyright",
    "©", "licensed under", "this work is",
)

_enc = tiktoken.get_encoding("cl100k_base")


def _extract_page_text(page) -> str:
    """Extract page text with proper word spacing.
    pdfplumber's default extract_text drops spaces on some PDFs (especially
    with non-standard glyph encodings). Reconstructing from extract_words
    is more reliable."""
    try:
        words = page.extract_words(use_text_flow=True)
    except Exception:
        return page.extract_text() or ""
    if not words:
        return page.extract_text() or ""

    # Bucket words by line using top-coordinate proximity.
    lines: dict[int, list] = {}
    for w in words:
        line_key = round(w["top"] / 3) * 3  # ~3px tolerance
        lines.setdefault(line_key, []).append(w)

    out_lines: list[str] = []
    for line_key in sorted(lines.keys()):
        sorted_words = sorted(lines[line_key], key=lambda w: w["x0"])
        out_lines.append(" ".join(w["text"] for w in sorted_words))
    return "\n".join(out_lines)


def _looks_like_attribution(line: str) -> bool:
    low = line.lower()
    return any(tok in low for tok in ATTRIBUTION_NOISE)


@dataclass
class Chunk:
    text: str
    paper_id: str
    paper_title: str
    section: str
    page: int
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            h = hashlib.sha1(f"{self.paper_id}|{self.section}|{self.page}|{self.text[:80]}".encode()).hexdigest()
            self.chunk_id = h[:16]


def _looks_like_heading(line: str) -> bool:
    """Heuristic: numbered or all-caps short lines that match known section names."""
    cleaned = re.sub(r"^[\d.\s]+", "", line).strip().lower()
    if len(cleaned) > 40:
        return False
    return cleaned in SECTION_HEADERS or any(cleaned.startswith(h) for h in SECTION_HEADERS)


def _extract_title(pdf: pdfplumber.PDF) -> str:
    """First substantial non-attribution line on page 1.
    Skips arxiv tags and copyright/attribution notices."""
    if not pdf.pages:
        return "Untitled"
    text = _extract_page_text(pdf.pages[0])
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 10 or len(line) > 200:
            continue
        if line.lower().startswith("arxiv"):
            continue
        if _looks_like_attribution(line):
            continue
        return line[:120]
    return "Untitled"


def _section_chunks(pdf: pdfplumber.PDF) -> list[tuple[str, int, str]]:
    """Return [(section_name, page, text)]. Falls back to fixed chunks if no sections detected."""
    sections: list[tuple[str, int, str]] = []
    current_section = "Preamble"
    current_page = 1
    buf: list[str] = []
    found_any_heading = False

    for page_num, page in enumerate(pdf.pages, start=1):
        text = _extract_page_text(page)
        for line in text.splitlines():
            if _looks_like_heading(line):
                if buf:
                    sections.append((current_section, current_page, "\n".join(buf).strip()))
                    buf = []
                current_section = re.sub(r"^[\d.\s]+", "", line).strip().title()
                current_page = page_num
                found_any_heading = True
            else:
                buf.append(line)
    if buf:
        sections.append((current_section, current_page, "\n".join(buf).strip()))

    if not found_any_heading:
        return _fallback_fixed_chunks(pdf)
    return [s for s in sections if len(s[2]) > 100]  # drop tiny fragments


def _fallback_fixed_chunks(pdf: pdfplumber.PDF) -> list[tuple[str, int, str]]:
    """When section detection fails: fixed 500-token chunks with overlap, page-tagged."""
    chunks: list[tuple[str, int, str]] = []
    for page_num, page in enumerate(pdf.pages, start=1):
        text = _extract_page_text(page).strip()
        if not text:
            continue
        tokens = _enc.encode(text)
        i = 0
        while i < len(tokens):
            window = tokens[i : i + FALLBACK_CHUNK_TOKENS]
            chunks.append(("Body", page_num, _enc.decode(window)))
            i += FALLBACK_CHUNK_TOKENS - FALLBACK_OVERLAP
    return chunks


def _split_oversized(section: str, page: int, text: str) -> list[tuple[str, int, str]]:
    """Split any chunk longer than MAX_CHUNK_TOKENS into overlapping sub-chunks.
    A long Methods section becomes Methods.1, Methods.2, etc. — same metadata, smaller pieces."""
    tokens = _enc.encode(text)
    if len(tokens) <= MAX_CHUNK_TOKENS:
        return [(section, page, text)]
    sub_chunks: list[tuple[str, int, str]] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + MAX_CHUNK_TOKENS]
        sub_chunks.append((section, page, _enc.decode(window)))
        i += MAX_CHUNK_TOKENS - FALLBACK_OVERLAP
    return sub_chunks


def parse_pdf(path: Path) -> list[Chunk]:
    """Parse a PDF into Chunks. Uses section-level chunking, falls back to fixed.
    Any chunk over MAX_CHUNK_TOKENS gets split into sub-chunks before embedding."""
    paper_id = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    with pdfplumber.open(path) as pdf:
        title = _extract_title(pdf)
        raw_sections = _section_chunks(pdf)

    # Apply hard token cap — long sections get split.
    sections: list[tuple[str, int, str]] = []
    for sec, pg, txt in raw_sections:
        sections.extend(_split_oversized(sec, pg, txt))

    return [
        Chunk(text=text, paper_id=paper_id, paper_title=title, section=section, page=page)
        for section, page, text in sections
    ]


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. OpenAI handles up to 2048 inputs per call."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest_pdf(path: Path, collection: chromadb.Collection, openai_client: OpenAI) -> dict:
    """End-to-end: parse, embed, upsert. Returns ingestion summary."""
    chunks = parse_pdf(path)
    if not chunks:
        return {"file": path.name, "chunks": 0, "title": "(empty)"}

    embeddings = embed_batch(openai_client, [c.text for c in chunks])

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "paper_id": c.paper_id,
                "paper_title": c.paper_title,
                "section": c.section,
                "page": c.page,
            }
            for c in chunks
        ],
    )
    return {"file": path.name, "chunks": len(chunks), "title": chunks[0].paper_title}
