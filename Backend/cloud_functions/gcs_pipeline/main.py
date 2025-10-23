"""Cloud Function entrypoint for GCS-triggered deal processing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from services.deal_pipeline import DealProcessingOrchestrator
from utils.firestore_utils import FirestoreManager


logger = logging.getLogger(__name__)

_pipeline = DealProcessingOrchestrator()
_firestore = FirestoreManager()


def handle_artifact_ingest(event: Dict[str, Any], context: Any) -> None:
    """Triggered on new files landing in the deals bucket."""

    bucket = event.get("bucket")
    name = event.get("name")
    if not bucket or not name:
        logger.warning("GCS trigger missing bucket/name; event=%s", event)
        return

    deal_id = _extract_deal_id(name)
    if not deal_id:
        logger.debug("Ignoring non-deal object: %s", name)
        return

    artifact_uri = f"gs://{bucket}/{name}"
    artifact_key = _infer_artifact_key(name)

    if artifact_key:
        asyncio.run(_update_raw_file(deal_id, artifact_key, artifact_uri))

    if artifact_key == "pitch_deck_url":
        logger.info("Starting processing pipeline for deal %s", deal_id)
        _pipeline.process(deal_id)


async def _update_raw_file(deal_id: str, key: str, uri: str) -> None:
    await _firestore.update_deal(deal_id, {f"raw_files.{key}": uri})


def _extract_deal_id(object_name: str) -> str | None:
    parts = object_name.split("/")
    if len(parts) < 2 or parts[0] != "deals":
        return None
    return parts[1]


def _infer_artifact_key(object_name: str) -> str | None:
    lower_name = object_name.lower()
    filename = lower_name.rsplit("/", 1)[-1]
    if filename.endswith(".pdf") and "pitch_deck" in filename:
        return "pitch_deck_url"
    if filename.endswith(".mp4"):
        return "video_pitch_deck_url"
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        return "audio_pitch_deck_url"
    if filename.endswith(".txt"):
        return "text_pitch_deck_url"
    return None
