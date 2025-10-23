"""Firestore trigger that loads memo JSON into BigQuery."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from google.cloud import bigquery

from config.settings import settings
from utils.vector_index import DealVectorIndex


logger = logging.getLogger(__name__)

_bigquery_client = bigquery.Client(project=settings.GCP_PROJECT_ID)
_dataset = settings.BIGQUERY_DATASET
_table = settings.BIGQUERY_MEMO_TABLE

try:
    _vector_index = DealVectorIndex()
except ValueError:
    _vector_index = None
except Exception as exc:  # pragma: no cover - external service init failures
    logger.warning("Vector index initialisation failed inside Cloud Function: %s", exc)
    _vector_index = None


def ingest_memo_to_bigquery(event: Dict[str, Any], context: Any) -> None:
    """Cloud Function entrypoint triggered by Firestore writes."""

    if not _dataset or not _table:
        logger.warning("BigQuery dataset/table not configured; skipping memo ingestion")
        return

    value = event.get("value") or {}
    fields = value.get("fields") or {}
    document = _decode_firestore_map(fields)

    memo_block = document.get("memo") or {}
    memo_payload = memo_block.get("draft_v1")
    if not memo_payload:
        logger.debug("Firestore update without memo payload; skipping BigQuery insert")
        return

    metadata = document.get("metadata") or {}
    deal_id = metadata.get("deal_id") or _extract_deal_id_from_context(context)
    if not deal_id:
        logger.warning("Unable to determine deal_id for BigQuery ingestion; skipping")
        return

    table_id = _build_table_id(_dataset, _table)

    row = {
        "deal_id": str(deal_id),
        "ingested_at": datetime.utcnow().isoformat(),
        "company_name": _safe_get(memo_payload, ["company_overview", "name"]),
        "sector": _safe_get(memo_payload, ["company_overview", "sector"]),
        "memo_json": json.dumps(memo_payload),
    }

    errors = _bigquery_client.insert_rows_json(table_id, [row])
    if errors:
        logger.error("BigQuery insertion errors: %s", errors)
    else:
        logger.info("Memo for deal %s inserted into %s", deal_id, table_id)

    if _vector_index:
        try:
            _vector_index.upsert_deal(str(deal_id), memo_payload, metadata)
        except Exception as exc:  # pragma: no cover - external service runtime failures
            logger.warning("Vector index update failed during ingestion: %s", exc)


def _decode_firestore_map(map_value: Dict[str, Any]) -> Dict[str, Any]:
    decoded: Dict[str, Any] = {}
    for key, value in map_value.items():
        decoded[key] = _decode_firestore_value(value)
    return decoded


def _decode_firestore_value(value: Dict[str, Any]) -> Any:
    if "nullValue" in value:
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "booleanValue" in value:
        return bool(value["booleanValue"])
    if "arrayValue" in value:
        return [
            _decode_firestore_value(item)
            for item in value["arrayValue"].get("values", [])
        ]
    if "mapValue" in value:
        return _decode_firestore_map(value["mapValue"].get("fields", {}))
    return value


def _safe_get(payload: Dict[str, Any], path: list[str]) -> str:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    if isinstance(current, str):
        return current
    if isinstance(current, (int, float)):
        return str(current)
    return ""


def _extract_deal_id_from_context(context: Any) -> str | None:
    if not context:
        return None
    resource = getattr(context, "resource", "")
    if not resource:
        return None
    parts = resource.split("/")
    try:
        deals_index = parts.index("documents") + 2
        return parts[deals_index]
    except (ValueError, IndexError):
        return None


def _build_table_id(dataset: str, table: str) -> str:
    if "." in table:
        return table
    if "." in dataset:
        return f"{dataset}.{table}"
    return f"{settings.GCP_PROJECT_ID}.{dataset}.{table}"
