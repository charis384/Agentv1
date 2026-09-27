"""ChromaDB query wrapper. Read-only: ingestion is handled by ingest.py."""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, INGEST_CFG


class KnowledgeBase:
    def __init__(self) -> None:
        # MUST be the exact same embedding model + collection used at ingestion time,
        # otherwise vectors live in different spaces and search returns garbage.
        embeddings = HuggingFaceEmbeddings(
            model_name=INGEST_CFG.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            collection_name=INGEST_CFG.collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._store._collection.count()  # noqa: SLF001

    def available_domains(self) -> list[str]:
        """Domains are discovered from the DB itself => new folders in /docs need zero code changes."""
        metas = self._store.get(include=["metadatas"]).get("metadatas") or []
        return sorted({m["domain"] for m in metas if m and "domain" in m})

    def search(self, queries: list[str], domain: str | None, k: int) -> list[Document]:
        """
        Multi-query search: run every query, merge, de-duplicate, keep the best k.
        A chunk matched by several queries keeps its best (lowest) cosine distance.
        """
        flt = {"domain": domain} if domain else None
        best: dict[tuple, tuple[float, Document]] = {}
        for q in queries:
            for doc, dist in self._store.similarity_search_with_score(q, k=k, filter=flt):
                m = doc.metadata
                key = (m.get("source"), m.get("page"), m.get("chunk_index"))
                if key not in best or dist < best[key][0]:
                    best[key] = (dist, doc)

        ranked = sorted(best.values(), key=lambda t: t[0])[:k]
        out = []
        for dist, doc in ranked:
            doc.metadata["distance"] = round(float(dist), 4)
            out.append(doc)
        return out


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """Singleton: loading the embedding model is slow, do it once per process."""
    return KnowledgeBase()
