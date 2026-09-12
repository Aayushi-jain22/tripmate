"""
Module 2 — Destination Knowledge Tool (RAG).

search_destination_guide(query: str) -> List[str]

Design decision: rather than pulling in a heavyweight vector DB + hosted
embedding API for a 4-city, ~20-chunk corpus, this uses a TF-IDF vector
space (scikit-learn) with cosine similarity as an in-memory "vector
store". It's deterministic, has zero external dependencies at query
time, and is easy for a reviewer to run offline. The retrieval
*interface* (`search_destination_guide`) is what the agent depends on,
so swapping this for FAISS/Chroma + real embeddings later is a
drop-in change — see README "Scaling the RAG tool".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "destination_data.json"


@dataclass(frozen=True)
class Chunk:
    city: str
    country: str
    section: str  # e.g. "visa", "packing_tips"
    text: str

    def as_result_string(self) -> str:
        return f"[{self.city} — {self.section}] {self.text}"


class RAGToolError(Exception):
    """Raised when the destination knowledge base can't be loaded or queried."""


class DestinationGuideRAG:
    def __init__(self, data_path: Path = DATA_PATH):
        self._chunks: List[Chunk] = self._load_chunks(data_path)
        # Index city/country/section alongside the body text so a query that
        # names the destination (but doesn't happen to repeat words from the
        # body text) still matches -- pure body-text TF-IDF under-recalls on
        # short queries like "customs in Thailand".
        self._corpus = [
            f"{c.city} {c.country} {c.section.replace('_', ' ')} {c.text}" for c in self._chunks
        ]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._corpus)
        self.known_cities = sorted({c.city for c in self._chunks})

    @staticmethod
    def _load_chunks(data_path: Path) -> List[Chunk]:
        if not data_path.exists():
            raise RAGToolError(f"Destination data pack not found at {data_path}")
        try:
            raw = json.loads(data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RAGToolError(f"Destination data pack is not valid JSON: {exc}") from exc

        chunks: List[Chunk] = []
        for city_entry in raw.get("cities", []):
            city = city_entry["city"]
            country = city_entry.get("country", "")
            for section, text in city_entry.get("sections", {}).items():
                chunks.append(Chunk(city=city, country=country, section=section, text=text))

        if not chunks:
            raise RAGToolError("Destination data pack loaded but contained zero chunks.")
        return chunks

    def search(self, query: str, top_k: int = 3, city_filter: str | None = None) -> List[str]:
        """Return the top_k most relevant chunk strings for a query.

        Raises RAGToolError on invalid input. Returns an empty list (not
        an exception) when nothing clears a minimal relevance bar --
        the caller/agent decides how to communicate "no info found".
        """
        if not query or not query.strip():
            raise RAGToolError("Query must be a non-empty string.")

        candidate_indices = range(len(self._chunks))
        if city_filter:
            candidate_indices = [
                i for i, c in enumerate(self._chunks)
                if c.city.lower() == city_filter.lower()
            ]
            if not candidate_indices:
                # Unknown city -- let the caller decide how to message this.
                return []

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix[candidate_indices])[0]

        ranked = sorted(zip(candidate_indices, sims), key=lambda x: x[1], reverse=True)
        top = [idx for idx, score in ranked[:top_k] if score > 0.05]
        return [self._chunks[i].as_result_string() for i in top]


# Module-level singleton so the (cheap) TF-IDF fit happens once per process.
_rag_instance: DestinationGuideRAG | None = None


def _get_instance() -> DestinationGuideRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = DestinationGuideRAG()
    return _rag_instance


def search_destination_guide(query: str, city_filter: str | None = None) -> List[str]:
    """Public tool function exposed to the agent."""
    return _get_instance().search(query, city_filter=city_filter)


def known_cities() -> List[str]:
    return _get_instance().known_cities
