# utils/ocr_utils.py

from typing import Dict, List, Tuple
import asyncio
import os
import json
import uuid
import logging
import time
from pathlib import Path

from google.cloud import vision
from google.cloud import storage
from PyPDF2 import PdfReader

from config.settings import settings
from utils.summarizer import GeminiSummarizer

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Vision OCR processor for PDFs stored in GCS.
    - Input: GCS path to PDF (e.g., "gs://<bucket>/deals/<id>/pitch_deck.pdf")
    - Output: Extracted page texts + concise Gemini summary
    """

    def __init__(self):
        # Summarizer (Gemini or local fallback)
        self.summarizer = GeminiSummarizer()

        self._use_vision = True
        self.vision_client = None
        self.storage_client = None

        try:
            self.vision_client = vision.ImageAnnotatorClient()
            self.storage_client = storage.Client()
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.warning("Google Vision unavailable (%s); using local PDF extraction", exc)
            self._use_vision = False

    async def process_pdf(self, gcs_path: str) -> Dict:
        """
        Orchestrates the OCR:
          1) Run Vision OCR (async) on the GCS PDF
          2) Read page texts from JSON outputs in GCS
          3) Summarize with Gemini
        """
        try:
            print("Process PDF Called")
            ocr_start = time.perf_counter()
            if self._use_vision and str(gcs_path).startswith("gs://"):
                page_texts, logo_candidates = await self._extract_text_from_pdf(gcs_path)
            else:
                page_texts, logo_candidates = await self._extract_text_locally(gcs_path)
            ocr_duration = time.perf_counter() - ocr_start

            # Join page texts to feed the summarizer
            full_text = "\n\n".join(
                f"Page {i + 1}: {t}" for i, t in enumerate(page_texts) if t
            )

            # print("full_text:",full_text)
            summary_start = time.perf_counter()
            concise_summary = await self.summarizer.summarize_pitch_deck(full_text)
            summary_duration = time.perf_counter() - summary_start
            # {"summary_res": response.text,
            #        "founder_response": founder_response,
            #        "sector_response": sector_response}
            print("concise_summary : ", concise_summary.get('founder_response'))
            print("concise_summary : ", concise_summary.get('sector_response'))

            logger.info(
                "PDF processing timings (s) for %s: {\"ocr\": %.3f, \"summarizer\": %.3f, \"pages\": %d}",
                gcs_path,
                ocr_duration,
                summary_duration,
                len(page_texts)
            )

            return {
                "raw": {str(i + 1): t for i, t in enumerate(page_texts)},
                "concise": concise_summary.get('summary_res', ''),
                "founder_response": concise_summary.get('founder_response', []),
                "sector_response": concise_summary.get('sector_response', ''),
                "company_name_response": concise_summary.get('company_name_response', ''),
                "product_name_response": concise_summary.get('product_name_response', ''),
                "logos": logo_candidates,
            }

        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            # Return a safe structure so callers don't break
            return {"raw": {"1": ""}, "concise": "Error in OCR processing."}

    async def _extract_text_locally(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Extract text from a local PDF without Google Vision."""

        def _read_pdf() -> List[str]:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"PDF not found at {file_path}")

            reader = PdfReader(path)
            texts: List[str] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
            return texts if texts else [""]

        texts = await asyncio.to_thread(_read_pdf)
        return texts, []

    async def _extract_text_from_pdf(self, gcs_path: str) -> Tuple[List[str], List[str]]:
        """
        Run Vision's AsyncBatchAnnotateFiles directly on the GCS PDF.
        Parse generated JSON outputs from GCS and return a list of page texts.
        """
        try:
            # ---- 1) Prepare Vision async request ----
            # Asynchronous file processing currently only supports
            # DOCUMENT_TEXT_DETECTION/TEXT_DETECTION features. Attempting to
            # request additional feature types (e.g. LOGO_DETECTION) results in
            # a 400 error: "only DOCUMENT_TEXT_DETECTION and TEXT_DETECTION are
            # supported". Request only text detection here and derive logos
            # separately if needed.
            text_feature = vision.Feature(
                type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
            )

            gcs_source = vision.GcsSource(uri=gcs_path)
            input_config = vision.InputConfig(
                gcs_source=gcs_source, mime_type="application/pdf"
            )

            # Keep outputs under a deterministic prefix to allow cleanup
            safe_name = os.path.basename(gcs_path).replace("/", "_")
            out_prefix = f"vision-output/{safe_name}/{uuid.uuid4().hex[:8]}"

            gcs_destination = vision.GcsDestination(
                uri=f"gs://{settings.GCS_BUCKET_NAME}/{out_prefix}/"
            )
            output_config = vision.OutputConfig(
                gcs_destination=gcs_destination,
                batch_size=5,  # tune if you want bigger chunks
            )

            async_request = vision.AsyncAnnotateFileRequest(
                features=[text_feature],
                input_config=input_config,
                output_config=output_config,
            )

            # ---- 2) Kick off async OCR and wait ----
            op = self.vision_client.async_batch_annotate_files(
                requests=[async_request]
            )
            # Up to 10 minutes for very large decks; tune as needed
            op.result(timeout=600)

            # ---- 3) Read JSON outputs back from GCS ----
            texts: List[str] = []
            logos_found: set[str] = set()

            bucket = self.storage_client.bucket(settings.GCS_BUCKET_NAME)
            # Materialize and sort for deterministic ordering
            blobs = sorted(
                list(bucket.list_blobs(prefix=out_prefix)),
                key=lambda b: b.name,
            )

            for blob in blobs:
                if not blob.name.endswith(".json"):
                    continue

                # Each JSON blob contains {"responses": [ ... ]} for one or more pages
                data = json.loads(blob.download_as_text())

                for resp in data.get("responses", []):
                    # Prefer the full text when present (per page)
                    full = resp.get("fullTextAnnotation", {})
                    page_text = full.get("text", "")

                    if not page_text:
                        # Fallback: reconstruct from words/symbols (rarely needed)
                        try:
                            reconstructed = []
                            for page in resp.get("pages", []):
                                for block in page.get("blocks", []):
                                    for para in block.get("paragraphs", []):
                                        words = []
                                        for word in para.get("words", []):
                                            symbols = "".join(
                                                s.get("text", "")
                                                for s in word.get("symbols", [])
                                            )
                                            words.append(symbols)
                                        reconstructed.append(" ".join(words))
                            page_text = "\n".join(reconstructed).strip()
                        except Exception:
                            page_text = ""

                    if page_text.strip():
                        texts.append(page_text.strip())

                    for logo in resp.get("logoAnnotations", []) or []:
                        description = str(logo.get("description", "")).strip()
                        if description:
                            logos_found.add(description)

                # Optional cleanup (keeps your bucket tidy)
                try:
                    blob.delete()
                except Exception:
                    pass

            # If nothing was extracted, keep a single empty string
            #print("text : ",texts)
            logo_list = sorted(logos_found)
            return (texts if texts else [""], logo_list)

        except Exception as e:
            logger.error(f"Vision API error: {str(e)}")
            # Return one empty string to keep callers safe
            return ([""], [])
