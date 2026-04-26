"""Dense retrieval using sentence-transformers embeddings.

Encodes each document with a small transformer model and retrieves by
cosine similarity. Forms the **dense-retrieval** half of the hybrid RAG
pipeline; combined with BM25 in `rag.py` via weighted score fusion.

Why MiniLM-L6-v2: 22 MB model, ~90 MB at runtime, encodes ~3000
sentences/sec on CPU, and produces a 384-d embedding that performs
competitively on MTEB retrieval benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


@dataclass
class DenseHit:
    docno: str
    headline: str
    body: str
    score: float


class DenseRetriever:
    """Sentence-transformer based dense retrieval over a folder of `.txt` docs."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers is required for DenseRetriever. "
                "Install with `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.docs: list[dict] = []
        self.embeddings: np.ndarray | None = None

    @classmethod
    def from_directory(
        cls, path: str | Path, model_name: str = "all-MiniLM-L6-v2"
    ) -> "DenseRetriever":
        retr = cls(model_name=model_name)
        retr.add_files(sorted(Path(path).glob("*.txt"), key=lambda x: x.name))
        return retr

    def add_files(self, paths: Iterable[Path]) -> None:
        docs = []
        texts = []
        for p in paths:
            text = p.read_text(encoding="utf-8")
            lines = [l for l in text.splitlines() if l.strip()]
            if not lines:
                continue
            headline = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            docs.append({"docno": p.stem, "headline": headline, "body": body})
            texts.append(f"{headline}\n{body}")
        if not texts:
            self.docs = []
            self.embeddings = None
            return
        self.docs = docs
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 10) -> list[DenseHit]:
        if self.embeddings is None or len(self.docs) == 0:
            return []
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ q_emb  # cosine — both sides are L2-normalised
        top_idx = np.argsort(-scores)[:top_k]
        return [
            DenseHit(
                docno=self.docs[i]["docno"],
                headline=self.docs[i]["headline"],
                body=self.docs[i]["body"],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def stats(self) -> dict:
        return {
            "documents": len(self.docs),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
            "model": self.model_name,
        }
