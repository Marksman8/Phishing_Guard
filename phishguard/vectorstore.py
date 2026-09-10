"""ChromaDB access.

Embeddings use all-MiniLM-L6-v2 via Chroma's bundled ONNX runtime rather than
sentence-transformers, which avoids a ~2GB PyTorch dependency for the identical
model. The collection is created with cosine space so similarity is 1 - distance
and scores are directly comparable across queries.
"""

import json
import logging
import os
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from . import config  # noqa: E402

# chromadb 0.5.x logs a posthog signature error on every call even with telemetry
# disabled; it is harmless and drowns out real output.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_CLIENT = None
_COLLECTION = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _CLIENT


def get_collection(create: bool = False):
    """Return the pattern collection, or None if it has not been ingested yet."""
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    client = get_client()
    try:
        if create:
            _COLLECTION = client.get_or_create_collection(
                name=config.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            _COLLECTION = client.get_collection(name=config.CHROMA_COLLECTION)
    except Exception:
        return None
    return _COLLECTION


def load_seed_patterns() -> list[dict[str, Any]]:
    if not config.SEED_PATTERNS.exists():
        return []
    rows = []
    for line in config.SEED_PATTERNS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def query(text: str, k: int = config.RETRIEVAL_K) -> list[dict[str, Any]]:
    """Nearest known patterns, with cosine similarity in [0, 1]. Returns [] when
    the index is missing so the caller can lower confidence honestly."""
    collection = get_collection()
    if collection is None or not text.strip():
        return []
    try:
        result = collection.query(
            query_texts=[text],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "label": (metadata or {}).get("label", "unknown"),
                "pattern": (metadata or {}).get("pattern", "unknown"),
                "source": (metadata or {}).get("source", "unknown"),
                "similarity": round(similarity, 4),
            }
        )
    return chunks


def collection_stats() -> dict[str, Any]:
    collection = get_collection()
    if collection is None:
        return {"available": False, "count": 0}
    try:
        return {"available": True, "count": collection.count()}
    except Exception:
        return {"available": False, "count": 0}
