"""Policy search for the airline agent: finds the FAQ sections most relevant to a question.
 This file only knows *how to search*; the `lookup_policy` tool
the bot actually calls lives in tools.py.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings


def split_faq(text: str) -> list[str]:
    """Split the FAQ markdown into sections, one per '##' heading ."""
    return [section.strip() for section in re.split(r"(?=\n##)", text) if section.strip()]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 so a dot product equals cosine similarity.
    A zero-length row is left as zeros instead of dividing by zero."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


def _load_cached_vectors(cache_path: str | None, cache_key: str, n_sections: int) -> np.ndarray | None:
    """Saved section vectors, or None if there is no usable cache (missing,
    unreadable, or built from a different FAQ / embedding model)."""
    if not cache_path or not Path(cache_path).exists():
        return None
    try:
        with np.load(cache_path) as cached:
            if str(cached["key"]) != cache_key or len(cached["vectors"]) != n_sections:
                return None
            return cached["vectors"]
    except (OSError, ValueError, KeyError):
        return None


class PolicyRetriever:
    def __init__(self, sections: list[str], vectors: np.ndarray, embedder: OllamaEmbeddings):
        self._sections = sections
        self._vectors = _normalize(vectors)
        self._embedder = embedder

    @classmethod
    def from_faq(cls, faq_path: str, embedding_model: str, ollama_host: str,
                 cache_path: str | None = None) -> "PolicyRetriever":
        """Read the FAQ file, split it into sections and embed every section once.

        With cache_path, the section vectors are saved to / loaded from that
        file (~1 min of CPU saved per run). The cache is only reused when it was
        built from exactly the same FAQ text and embedding model; otherwise it
        is rebuilt automatically.
        """
        text = Path(faq_path).read_text(encoding="utf-8")
        sections = split_faq(text)
        if not sections:
            raise ValueError(f"no policy sections found in {faq_path}")

        embedder = OllamaEmbeddings(model=embedding_model, base_url=ollama_host)
        cache_key = hashlib.sha256(f"{embedding_model}\n{text}".encode("utf-8")).hexdigest()

        vectors = _load_cached_vectors(cache_path, cache_key, len(sections))
        if vectors is None:
            vectors = np.array(embedder.embed_documents(sections))
            if cache_path:
                np.savez(cache_path, vectors=vectors, key=np.array(cache_key))
        return cls(sections, vectors, embedder)

    def query(self, question: str, k: int = 2) -> list[dict]:
        """Return the k sections most similar to the question, best first.

        Ties are broken by section order in the FAQ, so the same question
        always returns the same sections in the same order .
        """
        query_vector = _normalize(np.array(self._embedder.embed_query(question)))
        scores = self._vectors @ query_vector

        ranked = sorted(range(len(self._sections)), key=lambda i: (-scores[i], i))
        return [
            {"index": i, "score": round(float(scores[i]), 4), "section": self._sections[i]}
            for i in ranked[:k]
        ]
