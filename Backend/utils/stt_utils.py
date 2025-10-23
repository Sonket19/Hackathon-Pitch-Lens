import logging
from typing import Dict

from utils.summarizer import GeminiSummarizer

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio or video assets using Gemini multimodal capabilities."""

    def __init__(self) -> None:
        self.summarizer = GeminiSummarizer()

    async def process_audio(self, gcs_path: str) -> Dict:
        """Send raw audio to Gemini 1.5 Pro for direct analysis."""

        try:
            payload = await self.summarizer.analyze_media_from_gcs(
                gcs_path,
                mime_type=self._infer_audio_mime(gcs_path),
            )
            return payload
        except Exception as exc:
            logger.error("Audio processing error: %s", exc)
            raise

    async def process_video(self, gcs_path: str) -> Dict:
        """Stream a video asset to Gemini without intermediate transcription."""

        try:
            payload = await self.summarizer.analyze_media_from_gcs(
                gcs_path,
                mime_type=self._infer_video_mime(gcs_path),
            )
            return payload
        except Exception as exc:
            logger.error("Video processing error: %s", exc)
            raise

    @staticmethod
    def _infer_audio_mime(gcs_path: str) -> str:
        if gcs_path.endswith(".wav"):
            return "audio/wav"
        if gcs_path.endswith(".mp3"):
            return "audio/mpeg"
        if gcs_path.endswith(".aac"):
            return "audio/aac"
        return "audio/*"

    @staticmethod
    def _infer_video_mime(gcs_path: str) -> str:
        if gcs_path.endswith(".mp4"):
            return "video/mp4"
        if gcs_path.endswith(".mov"):
            return "video/quicktime"
        if gcs_path.endswith(".webm"):
            return "video/webm"
        return "video/*"
