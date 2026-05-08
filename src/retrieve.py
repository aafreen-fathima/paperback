"""Top-k retrieval from Chroma."""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from openai import OpenAI

from .ingest import EMBED_MODEL


@dataclass
class RetrievedChunk:
    text: str
    paper_title: str
    section: str
    page: int
    distance: float

    def citation_tag(self) -> str:
        # "Vaswani et al. 2017" style would need parsed metadata; for v0.1 use truncated title.
        short = self.paper_title.split(":")[0][:40]
        return f"[{short}, p.{self.page}]"


def retrieve(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    k: int = 6,
) -> list[RetrievedChunk]:
    """Embed query and pull top-k chunks. k=6 chosen empirically — see PRD."""
    q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding

    res = collection.query(query_embeddings=[q_emb], n_results=k)
    if not res["documents"] or not res["documents"][0]:
        return []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    return [
        RetrievedChunk(
            text=doc,
            paper_title=meta["paper_title"],
            section=meta["section"],
            page=int(meta["page"]),
            distance=float(dist),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]
