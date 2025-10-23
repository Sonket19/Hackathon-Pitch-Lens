from __future__ import annotations

import hashlib
import logging
from typing import Tuple

from fastapi import UploadFile
from google.cloud import storage

from config.settings import settings

logger = logging.getLogger(__name__)


class GCSManager:
    """Persist uploaded assets either to Google Cloud Storage or a local directory.

    The original implementation assumed that Google Cloud credentials were always
    available. In the hackathon environment that often isn't true, which meant the
    application crashed before the upload endpoint could respond.  We now attempt
    to initialise the GCS client but gracefully fall back to the filesystem when
    credentials or the bucket configuration are missing.  Callers can keep using
    the same API and will receive either a ``gs://`` path or an absolute local
    path depending on what was available at runtime.
    """

    def __init__(self):
        local_storage = getattr(settings, "LOCAL_STORAGE_PATH", "./storage")
        self._local_root = Path(local_storage).resolve()
        self._local_root.mkdir(parents=True, exist_ok=True)

    async def upload_file(self, file: UploadFile, destination_path: str) -> Tuple[str, str]:
        """Upload file to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(destination_path)

        if self._use_gcs:
            try:
                self.client = storage.Client(project=settings.GCP_PROJECT_ID)
                self.bucket = self.client.bucket(settings.GCS_BUCKET_NAME)
            except Exception as exc:  # pragma: no cover - depends on external creds
                logger.warning("GCS unavailable (%s); falling back to local storage", exc)
                self._use_gcs = False

            file_hash = hashlib.sha256(content).hexdigest()

            file_hash = hashlib.sha256(content).hexdigest()

            # Upload to GCS
            blob.upload_from_string(content, content_type=file.content_type)

        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()

            logger.info(f"File uploaded to GCS: {destination_path}")
            return f"gs://{settings.GCS_BUCKET_NAME}/{destination_path}", file_hash

        # Local fallback storage path
        local_path = (self._local_root / destination_path).resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        logger.info("File stored locally at %s", local_path)
        return str(local_path), file_hash

    async def download_file(self, gcs_path: str, local_path: str):
        """Download file from storage to a local path."""

        if gcs_path.startswith("gs://") and self._use_gcs and self.bucket:
            try:
                blob_name = gcs_path.replace(f"gs://{settings.GCS_BUCKET_NAME}/", "", 1)
                blob = self.bucket.blob(blob_name)
                blob.download_to_filename(local_path)
                logger.info("File downloaded from GCS: %s", gcs_path)
                return
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("Falling back to local copy for %s due to error: %s", gcs_path, exc)

        # Local files can simply be copied
        source = Path(gcs_path)
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        logger.info("File copied locally from %s to %s", source, destination)

    async def get_signed_url(self, gcs_path: str, expiration_minutes: int = 60) -> str:
        """Generate signed URL for private access.

        Signed URLs only make sense for real GCS objects.  When operating in
        local mode we simply return the absolute file path so the caller can
        access the asset directly.
        """

        if gcs_path.startswith("gs://") and self._use_gcs and self.bucket:
            try:
                blob_name = gcs_path.replace(f"gs://{settings.GCS_BUCKET_NAME}/", "", 1)
                blob = self.bucket.blob(blob_name)
                import datetime

                url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(minutes=expiration_minutes),
                    method="GET",
                )
                return url
            except Exception as exc:  # pragma: no cover - network failure
                logger.error("Signed URL generation failed for %s: %s", gcs_path, exc)

        return str(Path(gcs_path).resolve())
