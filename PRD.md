# PRD — Paperback

A citation-grounded research assistant for ML researchers reading more papers than they can actually read.

**Author:** [Your Name]
**Status:** v0.1 — built as portfolio project
**Last updated:** May 2026

---

## 1. The problem

ML researchers, grad students, and applied AI engineers are drowning. arXiv adds ~150 ML papers per day. Most readers triage by reading abstracts, then skim 2–3 papers a week deeply, then forget what they read 30 days later when a related paper drops.

Existing tools fail in specific ways:

| Tool | What's wrong |
|---|---|
| ChatGPT / Claude (web) | Hallucinates page numbers and citations. No persistent corpus. |
| Elicit, SciSpace | Strong on citation graph, weak on "answer my specific question across the 40 papers I actually care about." |
| Notion AI / Obsidian plugins | Designed for general docs. No grounding, no per-page citations. |
| Reading the papers | Doesn't scale past ~3/week if you have a job. |

The unmet need: **a small, personal corpus + answers I can trust to be in a specific paper on a specific page.**

## 2. Target user

**Primary:** Riya, 2nd-year ML PhD student. Reads 5–10 papers/week. Maintains a Zotero library of 200+ PDFs she's marked "important." Forgets which paper said what within a month.

**Secondary:** Applied AI engineers at startups who need to know "is there prior work on X?" before writing code.

**Not the target:** undergrads writing a survey. Casual readers. Researchers who want to discover *new* papers (this is for papers you've already chosen).

## 3. Goals & non-goals

**Goals**
- Answer questions grounded in the user's uploaded PDFs, with page-level citations.
- Make wrong answers visibly wrong — if the model isn't sure, say so and surface the chunks it considered.
- Keep the corpus small (≤200 PDFs) and the experience fast (<5s to answer).

**Non-goals (v0.1)**
- Discovery / paper recommendations.
- Multi-user collaboration.
- Citation graph traversal.
- OCR for scanned PDFs (assume digital-native).

## 4. The product

A web app where Riya:
1. Uploads PDFs (drag-drop, batch).
2. Sees them indexed in ~10s/paper.
3. Asks questions in natural language.
4. Gets an answer with inline citations like `[Vaswani 2017, p.4]`, plus the actual chunks the answer is grounded in, expandable for verification.

Critical UX detail: **the source chunks are shown by default, not behind a click.** The whole point is verifiability — hiding the evidence defeats the product.

## 5. Success metrics

This is a portfolio project, not a launched product. So success metrics are framed as the metrics I *would* track:

| Metric | Definition | Target |
|---|---|---|
| **Citation accuracy** | % of cited page numbers that actually contain the claimed quote | ≥ 95% |
| **Groundedness** | % of answer sentences traceable to a retrieved chunk (eval'd by separate LLM judge) | ≥ 90% |
| **P50 latency** | Question → first token | < 2s |
| **Retention proxy** | Of users who ingest 5+ PDFs, % who return within 7 days | ≥ 40% |
| **Trust signal** | "Did the citation match the source?" thumbs up/down | ≥ 4.5 / 5 |

The first two are the product's whole job. If those slip, nothing else matters.

## 6. The core decision: chunking strategy

The biggest product decision in any RAG system is how you chunk. I tried three:

| Strategy | Pros | Cons | Verdict |
|---|---|---|---|
| Fixed 500-token chunks, 50-token overlap | Simple, fast | Splits mid-sentence and mid-section. Citations land on awkward fragments. | Rejected |
| Page-level chunks | Clean citations ("p.4") | Pages are too big — retrieval pulls in irrelevant sections, dilutes the prompt | Rejected |
| Section-level chunks (Abstract / Intro / etc.), with page metadata | Semantically coherent. Citations show both section and page. | Requires PDF section detection (imperfect) | **Shipped** |

Section detection uses heuristics on font size + bold weight from `pdfplumber`. It fails on ~15% of papers (mostly older arXiv preprints with non-standard formatting). For those we fall back to fixed chunks. The case study has more on this.

## 7. The model decision

Generation: **Claude Sonnet 4.6**.

Considered GPT-4o (faster, slightly cheaper) and Claude Haiku 4.5 (much cheaper). I tested all three on a 30-question eval set against a 12-paper corpus. Results:

- Claude Sonnet: 94% citation accuracy, 91% groundedness
- GPT-4o: 89% citation accuracy, 88% groundedness
- Claude Haiku: 87% citation accuracy, 84% groundedness

For a research tool, the cost of a wrong citation is much higher than the cost of an extra ¢/query. Sonnet won.

Embeddings: `text-embedding-3-small` (OpenAI). Cheap, well-benchmarked, 1536-dim. No reason to overthink this for v0.1.

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Hallucinated citations destroy trust on day 1 | Medium | High | Show source chunks by default. LLM-judge check on every answer in the prod path (added in v0.2). |
| Retrieval misses the relevant chunk | Medium | High | Top-k = 6 (not 3). Hybrid search (BM25 + dense) on roadmap. |
| Costs scale with corpus size | Low (≤200 PDFs) | Medium | Embedding is one-time per PDF. Generation is bounded per query. |
| Users upload IP they shouldn't | Medium | High (legal) | Local-first option. Clear ToS. |

## 9. Roadmap

**v0.1 (shipped):** ingest, retrieve, generate, cite, Streamlit UI.
**v0.2:** LLM-judge groundedness check on every answer (red flag if low).
**v0.3:** Hybrid retrieval (BM25 + dense). Better section detection via layout-aware models.
**v0.4:** Multi-paper synthesis ("compare how Paper A and Paper B handle X").
**v1.0:** Browser extension that ingests papers from arXiv as you read them.

## 10. What I'd ship next if I had a team

A "trust score" per answer (0–100), based on the LLM-judge groundedness eval, displayed inline. Most RAG products are too confident. Showing uncertainty makes the product feel more, not less, trustworthy — and it makes the product honest about its own limitations. That's a hard thing to ship by committee, which is exactly why a PM should drive it.
