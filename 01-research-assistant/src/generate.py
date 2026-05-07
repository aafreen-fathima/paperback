"""Prompt construction and Claude call."""
from __future__ import annotations

from anthropic import Anthropic

from .retrieve import RetrievedChunk

GEN_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a research assistant. You answer questions strictly based on the provided source chunks from academic papers.

Rules:
1. Cite every claim inline using the format [Paper Title, p.N]. Use the exact citation tags shown next to each chunk.
2. If the sources don't contain the answer, say so explicitly. Do NOT use outside knowledge.
3. Quote sparingly — paraphrase and cite. Quote only when the exact phrasing matters.
4. If sources contradict each other, surface that contradiction rather than picking one.
5. Be concise. Researchers don't want padding — they want the answer and the citation."""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Construct the user-side prompt with numbered chunks."""
    if not chunks:
        return f"Question: {question}\n\nNo source chunks were retrieved. Answer: (none — no sources available)"

    sources_block = "\n\n".join(
        f"--- Source {i+1} {c.citation_tag()} (section: {c.section}) ---\n{c.text}"
        for i, c in enumerate(chunks)
    )

    return (
        f"Question: {question}\n\n"
        f"Source chunks:\n{sources_block}\n\n"
        f"Answer the question using only the sources above. Cite inline."
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    anthropic_client: Anthropic,
) -> str:
    """Single Claude call. Streaming would be nice; not needed for v0.1."""
    user = build_user_prompt(question, chunks)
    resp = anthropic_client.messages.create(
        model=GEN_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    # Defensive: response shape is a list of blocks; we want the text.
    return "".join(block.text for block in resp.content if block.type == "text")
