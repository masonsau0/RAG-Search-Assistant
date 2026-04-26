# RAG Search Assistant

A **retrieval-augmented generation (RAG)** system that answers natural-language
questions over a company knowledge base. Combines **BM25** (sparse,
term-overlap) and **sentence-transformer embeddings** (dense, semantic)
through weighted score fusion, then sends the top-K retrieved chunks to a
**large language model (LLM)** with a strict "only use the context"
prompt — returning a grounded answer with numbered source citations.

The bundled demo corpus is **37 short policy / FAQ documents** modelled on
a typical tech-company internal knowledge base — HR & benefits, engineering
practices, security, onboarding, office logistics, and internal tools — so
the assistant runs out of the box.

## What it actually does

1. **Indexes** the corpus twice at startup:
   - A pure-Python **BM25 inverted index** for sparse, term-overlap retrieval.
   - A **sentence-transformer** (`all-MiniLM-L6-v2`, 22 MB) that encodes every
     document into a 384-dimensional vector for dense, semantic retrieval.
2. On each query, **runs both retrievers** in parallel and **fuses the
   scores** via min-max normalisation followed by a weighted sum:
   `hybrid = α · bm25_norm + (1 − α) · dense_norm`. The α slider in the
   dashboard exposes this trade-off live.
3. Builds a **grounded prompt** that includes the top-K retrieved chunks with
   numbered citations and instructs the LLM to (a) use only the provided
   context, (b) cite sources by their bracket number, (c) say so directly if
   the answer is not in the context.
4. **Calls the LLM** — Anthropic Claude (preferred) or OpenAI GPT (fallback)
   — and returns the answer alongside the retrieved chunks and the exact
   prompt that was sent, so grounding is auditable.

If neither API key is set, the engine runs in **extractive-fallback mode** —
the retrieval pipeline still works end-to-end and you can see the prompt
that would have been sent.

## Why hybrid retrieval

Pure BM25 misses paraphrases — searching for "how do I take time off" against
a corpus that uses the term "PTO" returns nothing useful. Pure dense retrieval
misses exact-keyword matches — a query for a specific tool name gets
out-ranked by semantically-similar but irrelevant documents. The two methods
fail in opposite ways, and a weighted fusion captures the best of both:

| Query style | BM25 wins | Dense wins |
|---|---|---|
| Exact keywords / acronyms | ✓ | |
| Paraphrases / synonyms | | ✓ |
| Long, descriptive questions | partial | ✓ |
| Out-of-vocabulary terms | | ✓ |

The α slider lets you see this on real queries against the demo corpus.

## Repository layout

```
.
├── rag.py                  ← hybrid retrieval + LLM call orchestration
├── bm25.py                 ← from-scratch BM25 sparse retrieval
├── dense_retriever.py      ← sentence-transformer dense retrieval
├── rag_app.py              ← Streamlit dashboard
├── build_demo_corpus.py    ← generates the 37-doc demo knowledge base
├── corpus/                 ← 37 .txt policy / FAQ documents
├── requirements.txt
├── LICENSE
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
python build_demo_corpus.py        # writes corpus/*.txt (37 documents)
streamlit run rag_app.py
```

The first launch downloads the `all-MiniLM-L6-v2` model (~22 MB) and
indexes the corpus — subsequent launches are cached.

### Enabling the LLM

Set one of the following environment variables before launching:

```bash
# Anthropic (preferred — uses claude-haiku-4-5)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI fallback (uses gpt-4o-mini)
export OPENAI_API_KEY=sk-...
```

On Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
streamlit run rag_app.py
```

Without an API key, the dashboard runs in **extractive-fallback mode** —
retrieval still works end-to-end, and the exact prompt that would be sent
to the LLM is visible in the "Show prompt" panel.

## Programmatic use

```python
from rag import RAGEngine

engine = RAGEngine(corpus_path="corpus")
result = engine.answer("How do I request paid time off?", top_k=5, alpha=0.5)

print(result.answer)
for c in result.contexts:
    print(f"[{c.docno}] {c.headline} — hybrid={c.hybrid_score:.3f}")
```

## Dashboard features

- **α slider** (0.0 – 1.0) — live blend between BM25 (α = 1) and dense (α = 0).
  Pick a query and slide α from 1 to 0 to watch the result set shift from
  keyword-matching to semantic-similar.
- **Top-K slider** (1 – 10) — how many chunks to feed to the LLM.
- **Per-result score breakdown** — every retrieved chunk shows its hybrid,
  BM25-normalised, and dense-normalised scores side by side.
- **Prompt visibility** — the exact prompt sent to the LLM is shown,
  citations and all, so the grounding is auditable.
- **Provider auto-detection** — sidebar reports which LLM is in use, or
  warns when neither key is set.

## Stack

- **Python** (no third-party retrieval/IR library — BM25 is implemented
  from scratch)
- **sentence-transformers** (`all-MiniLM-L6-v2`) for dense embeddings
- **NumPy** for vectorised cosine similarity
- **Anthropic / OpenAI** Python SDKs for LLM calls
- **Streamlit** for the interactive dashboard
