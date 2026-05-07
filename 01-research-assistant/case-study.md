# Case Study — Paperback (RAG Research Assistant)

**One-line:** A citation-grounded research assistant for ML researchers — built in a week, shipped on Streamlit Cloud, evaluated against a 30-question test set.

**Live demo:** [streamlit-cloud-link-here] *(replace with your deploy URL)*
**Code:** see `src/`
**PRD:** see `PRD.md`

---

## What this is

Most "RAG demos" stop at *can the LLM answer using my docs?* That's the easy part. The hard part is making the answer trustworthy — citations that point to real pages, sources visible by default, and a graceful failure when retrieval misses.

I built Paperback to push past the demo bar. v0.1 has working ingestion (PDF → chunks → embeddings → Chroma), retrieval (top-k cosine), generation (Claude Sonnet), and a Streamlit UI with inline citations and source chunk previews.

## The interesting decisions

### 1. I rejected page-level chunking, even though it gave the cleanest citations

The first thing you want when building a RAG citation system is: chunk = page. Then your citation is just the page number. Beautiful, simple.

But page-level chunks are 500–1000 tokens of *unrelated* content. The intro, a figure, a forward-reference, a footnote, all jammed into one chunk. Cosine similarity averages over all of that, which means retrieval pulls in irrelevant pages because the relevant sentence got diluted.

Section-level chunking is messier (you have to actually detect sections from PDF layout) but the retrieval is dramatically better. Citations get a section name *and* a page number, which is actually more useful than just a page.

**PM lesson:** the "obvious" architecture is often the one optimized for the engineer, not the user. The user wants to find the answer, not have a tidy data model.

### 2. I built the eval set before I built the product

Before writing any retrieval code, I wrote 30 questions across a 12-paper corpus, with the exact (paper, page, quote) where the answer lives. This was 4 hours of work and it was the highest-leverage thing I did.

It told me:
- Which chunking strategy actually wins (objective answer, not vibes).
- When my retrieval was broken (recall@6 dropped from 87% → 71% when I increased chunk size).
- When upgrading the model was worth it (Sonnet > GPT-4o by a real margin on this corpus).

This is what becomes Project 3.

### 3. I chose not to use LangChain

LangChain wraps everything in abstractions you have to learn anyway. For a 200-line RAG pipeline, raw `chromadb` + `openai` + `anthropic` SDKs were faster to write *and* easier to debug. The chain abstraction also obscures the actual prompt — which is the most important thing in any LLM product.

I'd reconsider this for a multi-step agent (Project 2), where the orchestration logic is the hard part.

## What I cut from v0.1

- **Hybrid retrieval (BM25 + dense).** Worth ~3% on recall in my tests, but the integration cost was 2 days. Pushed to v0.3.
- **Streaming responses.** Looks better, but the bottleneck is retrieval (~1s) not generation. Felt like polish for a portfolio piece, not core.
- **Multi-corpus support.** Every user has one corpus in v0.1. Adding workspaces would be 80% of the remaining UX work.

## What broke / what I'd do differently

The PDF section detection fails on ~15% of papers. My current fallback is "use fixed 500-token chunks for those." The right answer is probably to use a layout-aware model (something like LayoutLMv3 or a small VLM) to detect sections from rendering, not text. Didn't have time to plumb this in.

If I were building this for real, I'd also instrument **per-question groundedness** as a runtime check, not just an offline eval. Right now I trust the model to be grounded; I should be verifying it on every answer.

## What this would look like as a real product

If a startup hired me to take this from v0.1 to launchable:

**Months 1–2:** Stand up the eval harness as a CI gate. Every model swap, prompt change, retrieval change → re-runs the eval set, blocks merge if groundedness drops > 2%. Build the per-answer groundedness runtime check.

**Months 3–4:** User research with 8–10 ML PhD students. The UX hypothesis I'd test: do people actually want a chat interface, or do they want question → answer with no thread? My intuition says no-thread, but I'd want to see it before committing.

**Months 5–6:** Multi-corpus support, browser extension for arXiv ingestion, public beta.

The product moat isn't the RAG — anyone can build that now. The moat is the trust system: citation accuracy, groundedness, and the UX that makes both visible.

## What I learned

1. **Eval > intuition.** I changed my mind about chunking strategy three times based on data, not opinion.
2. **The PRD is the product, twice.** Writing it forced me to be honest about what wasn't in scope.
3. **RAG is a UX problem more than a retrieval problem.** The hardest decisions aren't "which embedding model" — they're "do we show source chunks by default?"

---

*Built May 2026. Open to feedback — DM me on LinkedIn.*
