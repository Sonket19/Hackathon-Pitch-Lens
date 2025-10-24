from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Tuple
from io import BytesIO

from fastapi import UploadFile
from google.cloud import storage

from config.settings import settings

logger = logging.getLogger(__name__)


class GCSManager:
    def __init__(self):
        self.client = storage.Client(project=settings.GCP_PROJECT_ID)
        self.bucket = self.client.bucket(settings.GCS_BUCKET_NAME)

    async def upload_file(self, file: UploadFile, destination_path: str) -> Tuple[str, str]:
        """Upload file from FastAPI UploadFile to GCS"""
        try:
            blob = self.bucket.blob(destination_path)
            content = await file.read()
            file_hash = hashlib.sha256(content).hexdigest()

            # upload_from_string is synchronous
            blob.upload_from_string(content, content_type=file.content_type)

            logger.info(f"File uploaded to GCS: {destination_path}")
            return f"gs://{settings.GCS_BUCKET_NAME}/{destination_path}", file_hash

        except Exception as e:
            logger.error(f"GCS upload error: {str(e)}")
            raise

    def upload_blob_from_bytes(
        self,
        data: bytes,
        destination_blob_name: str,
        content_type: str = "application/pdf",
    ) -> str:
        """Uploads raw bytes to GCS and returns the resulting ``gs://`` URI."""
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_string(data, content_type=content_type)
            logger.info(f"Bytes uploaded to GCS: {destination_blob_name}")
            return f"gs://{self.bucket.name}/{destination_blob_name}"
        except Exception as e:
            logger.error(f"GCS bytes upload error: {str(e)}")
            raise

    def download_blob(self, blob_name: str) -> bytes:
        """Downloads a blob from GCS as bytes."""
        try:
            blob = self.bucket.blob(blob_name)
            content = blob.download_as_bytes()
            logger.info(f"File downloaded from GCS: {blob_name}")
            return content
        except Exception as e:
            logger.error(f"GCS download error: {str(e)}")
            raise

    def delete_blob(self, blob_name: str):
        """Deletes a blob from GCS."""
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"Blob deleted from GCS: {blob_name}")
        except Exception as e:
            logger.error(f"GCS delete error: {str(e)}")
            # Don't raise, just log warning
            logger.warning(f"Failed to delete blob {blob_name}: {e}")

    # This function seems unused in the orchestrator, but leaving it
    def download_file(self, gcs_path: str, local_path: str):
        """Download file from GCS to local path"""
        try:
            blob_name = gcs_path.replace(f"gs://{settings.GCS_BUCKET_NAME}/", "")
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            logger.info(f"File downloaded from GCS: {gcs_path}")
        except Exception as e:
            logger.error(f"GCS download error: {str(e)}")
            raise


# Instantiate a single manager to be imported
gcs_manager = GCSManager()
