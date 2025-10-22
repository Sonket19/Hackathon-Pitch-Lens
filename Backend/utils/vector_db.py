"""Utility helpers for interacting with Vertex AI Vector Search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

try:  # pragma: no cover - optional dependency across environments
    from vertexai.preview.vector_search import VectorSearchClient
except ImportError:  # pragma: no cover - fallback when preview client unavailable
    VectorSearchClient = None  # type: ignore[assignment]

from config.settings import settings

logger = logging.getLogger(__name__)


_EMBEDDING_MODEL: Optional[TextEmbeddingModel] = None
_VECTOR_CLIENT: Optional[VectorSearchClient] = None


def _ensure_vertex_initialised() -> None:
    try:
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    except ValueError:  # pragma: no cover - already initialised
        logger.debug("Vertex initialisation already performed", exc_info=True)


def _embedding_model() -> TextEmbeddingModel:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _ensure_vertex_initialised()
        _EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return _EMBEDDING_MODEL


def _vector_client() -> VectorSearchClient:
    global _VECTOR_CLIENT
    if VectorSearchClient is None:
        raise RuntimeError("vertexai.preview.vector_search is unavailable in this environment")
    if _VECTOR_CLIENT is None:
        _ensure_vertex_initialised()
        _VECTOR_CLIENT = VectorSearchClient(
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
    return _VECTOR_CLIENT


def chunk_document(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks for embedding."""

    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def generate_embeddings(texts: Sequence[str]) -> List[List[float]]:
    """Embed a batch of texts using text-embedding-004."""

    if not texts:
        return []

    model = _embedding_model()
    responses = model.get_embeddings(texts)
    return [list(record.values) for record in responses]


@dataclass
class _VectorConfig:
    index: str
    endpoint: str
    deployed_id: str


def _config() -> Optional[_VectorConfig]:
    if not settings.VECTOR_SEARCH_INDEX or not settings.VECTOR_SEARCH_INDEX_ENDPOINT or not settings.VECTOR_SEARCH_DEPLOYED_INDEX_ID:
        logger.warning("Vector search configuration missing; skipping operation")
        return None
    return _VectorConfig(
        index=settings.VECTOR_SEARCH_INDEX,
        endpoint=settings.VECTOR_SEARCH_INDEX_ENDPOINT,
        deployed_id=settings.VECTOR_SEARCH_DEPLOYED_INDEX_ID,
    )


def upsert_documents(
    documents: Sequence[str],
    *,
    namespace: str = "general",
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Embed and upsert documents into the configured vector index."""

    cfg = _config()
    if cfg is None:
        return

    metadatas = list(metadatas or [])
    embeddings = generate_embeddings(documents)
    datapoints = []

    for idx, embedding in enumerate(embeddings):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        if metadata is None:
            metadata = {}
        metadata = {**metadata, "namespace": namespace}
        datapoints.append(
            {
                "datapoint_id": metadata.get("id") or f"{namespace}-{idx}",
                "feature_vector": embedding,
                "crowding_tag": namespace,
                "restricts": [],
                "attributes": metadata,
            }
        )

    if not datapoints:
        return

    try:
        client = _vector_client()
    except RuntimeError as exc:  # pragma: no cover - dependency missing
        logger.warning("Skipping vector upsert: %s", exc)
        return
    try:
        client.upsert_datapoints(
            index=cfg.index,
            datapoints=datapoints,
        )
    except Exception as exc:  # pragma: no cover - network/service failure
        logger.error("Failed to upsert datapoints: %s", exc)


def query_index(query: str, *, source: str = "general", k: int = 5) -> List[Dict[str, Any]]:
    """Query the deployed vector index for nearest neighbours."""

    cfg = _config()
    if cfg is None or not query:
        return []

    embedding = generate_embeddings([f"{source}: {query}"])
    if not embedding:
        return []

    try:
        client = _vector_client()
    except RuntimeError as exc:  # pragma: no cover - dependency missing
        logger.warning("Vector search client unavailable: %s", exc)
        return []
    try:
        response = client.find_neighbors(
            index_endpoint=cfg.endpoint,
            deployed_index_id=cfg.deployed_id,
            queries=[
                {
                    "datapoint_id": "query",
                    "feature_vector": embedding[0],
                    "neighbor_count": k,
                    "filter": {"namespace": source},
                }
            ],
        )
    except Exception as exc:  # pragma: no cover - network/service failure
        logger.error("Vector search query failed: %s", exc)
        return []

    neighbors = []
    for match in getattr(response, "nearest_neighbors", []) or []:
        for neighbor in getattr(match, "neighbors", []) or []:
            neighbors.append(
                {
                    "datapoint_id": getattr(neighbor, "datapoint_id", None),
                    "distance": getattr(neighbor, "distance", None),
                    "metadata": getattr(neighbor, "attributes", None) or {},
                    "score": getattr(neighbor, "distance", None),
                    "text": getattr(neighbor, "attributes", {}).get("text"),
                    "title": getattr(neighbor, "attributes", {}).get("title"),
                    "link": getattr(neighbor, "attributes", {}).get("url"),
                    "snippet": getattr(neighbor, "attributes", {}).get("snippet"),
                }
            )

    # Some client libraries return raw dicts instead of objects; handle that gracefully.
    if not neighbors and isinstance(response, dict):  # pragma: no cover - alternate response format
        raw_matches = response.get("nearestNeighbors") or []
        for match in raw_matches:
            for neighbor in match.get("neighbors", []):
                metadata = neighbor.get("attributes") or {}
                neighbors.append(
                    {
                        "datapoint_id": neighbor.get("datapoint_id"),
                        "distance": neighbor.get("distance"),
                        "metadata": metadata,
                        "score": neighbor.get("distance"),
                        "text": metadata.get("text"),
                        "title": metadata.get("title"),
                        "link": metadata.get("url"),
                        "snippet": metadata.get("snippet"),
                    }
                )

    return neighbors
