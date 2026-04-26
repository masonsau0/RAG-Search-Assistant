"""Hybrid retrieval + LLM-grounded answer generation.

Combines BM25 (sparse, term-overlap) and sentence-transformer embeddings
(dense, semantic) via weighted score fusion, then sends the top-K
contexts to an LLM with a strict "only-use-the-context" prompt. Returns
the answer alongside the retrieved chunks and the exact prompt that was
sent — so the user can audit grounding.

LLM provider selection: ANTHROPIC_API_KEY is preferred, OPENAI_API_KEY is
the fallback. Without either, the engine returns retrieval results in an
extractive-fallback mode that still demonstrates the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bm25 import Corpus
from dense_retriever import DenseRetriever


@dataclass
class RAGContext:
    docno: str
    headline: str
    body: str
    hybrid_score: float
    bm25_score: float
    dense_score: float


@dataclass
class RAGResult:
    query: str
    answer: str
    contexts: list[RAGContext]
    prompt: str
    provider: str


class RAGEngine:
    """Hybrid BM25 + dense retrieval, then LLM-grounded answer generation."""

    def __init__(
        self,
        corpus_path: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.corpus_path = Path(corpus_path)
        self.bm25 = Corpus.from_directory(self.corpus_path, k1=k1, b=b)
        self.dense = DenseRetriever.from_directory(
            self.corpus_path, model_name=embedding_model
        )

    def retrieve(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[RAGContext]:
        """Hybrid retrieval. alpha=1 → BM25 only; alpha=0 → dense only."""
        bm25_hits = self.bm25.search(query, top_k=top_k * 3)
        dense_hits = self.dense.search(query, top_k=top_k * 3)

        # Min-max normalise within each retriever so weighted fusion is meaningful.
        bm25_max = max((h.score for h in bm25_hits), default=1.0) or 1.0
        dense_max = max((h.score for h in dense_hits), default=1.0) or 1.0
        bm25_norm = {h.docno: h.score / bm25_max for h in bm25_hits}
        dense_norm = {h.docno: h.score / dense_max for h in dense_hits}

        all_docnos = set(bm25_norm) | set(dense_norm)
        scored = []
        for docno in all_docnos:
            bs = bm25_norm.get(docno, 0.0)
            ds = dense_norm.get(docno, 0.0)
            score = alpha * bs + (1 - alpha) * ds
            scored.append((docno, score, bs, ds))
        scored.sort(key=lambda x: -x[1])

        results = []
        for docno, hybrid, bs, ds in scored[:top_k]:
            doc_id = self.bm25.docno_to_id.get(docno)
            if doc_id is None:
                continue
            doc = self.bm25.docs[doc_id]
            results.append(RAGContext(
                docno=docno,
                headline=doc.headline,
                body=doc.body,
                hybrid_score=float(hybrid),
                bm25_score=float(bs),
                dense_score=float(ds),
            ))
        return results

    def answer(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        max_context_chars: int = 4000,
    ) -> RAGResult:
        contexts = self.retrieve(query, top_k=top_k, alpha=alpha)
        prompt = self._build_prompt(query, contexts, max_context_chars)
        answer, provider = self._call_llm(prompt)
        return RAGResult(
            query=query, answer=answer, contexts=contexts,
            prompt=prompt, provider=provider,
        )

    def _build_prompt(
        self, query: str, contexts: list[RAGContext], max_chars: int = 4000
    ) -> str:
        ctx_blocks = []
        running = 0
        for i, c in enumerate(contexts, 1):
            block = f"[{i}] {c.headline}\n{c.body}\n"
            if running + len(block) > max_chars:
                break
            ctx_blocks.append(block)
            running += len(block)
        ctx = "\n".join(ctx_blocks)
        return (
            "You are a helpful assistant answering questions using only the company "
            "knowledge base below.\n\n"
            "Rules:\n"
            "- Only use information present in the context. If the answer is not "
            "in the context, say so directly.\n"
            "- Cite the source by the bracketed number, e.g. [1] or [2].\n"
            "- Keep the answer concise and factual.\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

    def _call_llm(self, prompt: str) -> tuple[str, str]:
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                client = anthropic.Anthropic()
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text, "anthropic (claude-haiku-4-5)"
            except Exception as e:
                return f"[Anthropic call failed: {e}]", "anthropic (failed)"

        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                return resp.choices[0].message.content, "openai (gpt-4o-mini)"
            except Exception as e:
                return f"[OpenAI call failed: {e}]", "openai (failed)"

        return (
            "[No LLM API key found - showing extractive-fallback mode]\n\n"
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable grounded "
            "LLM answer generation. The retrieved contexts shown above contain "
            "the source material that would be sent to the LLM, formatted in "
            "the exact prompt visible in the 'Show prompt sent to LLM' panel.",
            "extractive (no LLM key set)",
        )

    def stats(self) -> dict:
        bm25_stats = self.bm25.stats()
        dense_stats = self.dense.stats()
        return {
            "documents": bm25_stats["documents"],
            "vocabulary": bm25_stats["vocabulary"],
            "total_tokens": bm25_stats["total_tokens"],
            "avg_doc_length": bm25_stats["avg_doc_length"],
            "embedding_dim": dense_stats["embedding_dim"],
            "embedding_model": dense_stats["model"],
        }
