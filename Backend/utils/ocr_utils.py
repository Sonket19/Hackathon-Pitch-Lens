import io
import logging
from typing import List, Sequence, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - dependency availability is environment-specific
    from pypdf import PdfReader, PdfWriter
except ModuleNotFoundError:  # pragma: no cover - allows unit tests without optional dependency
    PdfReader = None  # type: ignore[assignment]
    PdfWriter = None  # type: ignore[assignment]

from google.cloud import documentai_v1 as documentai

logger = logging.getLogger(__name__)

PAGE_LIMIT = 15  # The hard quota for standard Document AI OCR

ChunkRange = Tuple[int, int]


class DocumentAIProcessingError(RuntimeError):
    """Raised when Document AI fails to return text for a given request."""


class DocumentAIPageLimitError(DocumentAIProcessingError):
    """Raised when Document AI rejects a request because of page limits."""


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """Parses a GCS URI into bucket and blob name."""
    parsed_uri = urlparse(gcs_uri)
    if parsed_uri.scheme != "gs":
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name = parsed_uri.netloc
    blob_name = parsed_uri.path.lstrip('/')
    return bucket_name, blob_name


def calculate_page_chunks(total_pages: int, page_limit: int = PAGE_LIMIT) -> List[ChunkRange]:
    """Return half-open page ranges that respect the provided ``page_limit``."""
    if total_pages < 0:
        raise ValueError("total_pages cannot be negative")
    if page_limit <= 0:
        raise ValueError("page_limit must be a positive integer")

    return [
        (start_page, min(start_page + page_limit, total_pages))
        for start_page in range(0, total_pages, page_limit)
    ]


def extract_text_from_pdf_docai(
    gcs_uri: str,
    project_id: str | None,
    location: str | None,
    processor_id: str | None,
    *,
    client: documentai.DocumentProcessorServiceClient | None = None,
    processor_resource: str | None = None,
) -> str:
    """Process a single document chunk (<=15 pages) with Document AI."""

    if client is None:
        if not (project_id and location and processor_id):
            raise ValueError(
                "project_id, location, and processor_id are required when no client is provided."
            )
        client_options = {"api_endpoint": f"{location}-documentai.googleapis.com"}
        client = documentai.DocumentProcessorServiceClient(client_options=client_options)
        processor_resource = client.processor_path(project_id, location, processor_id)
        manage_client = True
    else:
        if not processor_resource:
            raise ValueError("processor_resource must be provided when supplying a client instance")
        manage_client = False

    logger.info("Starting Document AI processing for chunk: %s", gcs_uri)

    gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type="application/pdf")
    request = documentai.ProcessRequest(
        name=processor_resource,
        gcs_document=gcs_document,
        skip_human_review=True,
    )

    try:
        result = client.process_document(request=request)
        document = result.document
        logger.info("Document AI processing complete for chunk: %s", gcs_uri)
        return document.text
    except Exception as exc:  # pragma: no cover - network/service failure path
        error_message = str(exc)

        if "PAGE_LIMIT_EXCEEDED" in error_message:
            logger.error(
                "Document AI reported PAGE_LIMIT_EXCEEDED for chunk %s: %s",
                gcs_uri,
                error_message,
            )
            raise DocumentAIPageLimitError(error_message) from exc

        logger.error("Error in Document AI processing chunk %s: %s", gcs_uri, error_message)
        raise DocumentAIProcessingError(error_message) from exc
    finally:  # pragma: no cover - exercised in integration
        if manage_client:
            try:
                client.transport.close()
            except AttributeError:
                pass


async def process_large_pdf(
    gcs_uri: str,
    deal_id: str,
    project_id: str,
    location: str,
    processor_id: str,
) -> str:
    """Process large PDFs by splitting them into chunks that respect the 15-page limit."""

    from .gcs_utils import gcs_manager  # Local import to avoid heavy initialization during tests

    logger.info("Starting large PDF processing for %s", gcs_uri)

    if PdfReader is None or PdfWriter is None:
        logger.error("pypdf is required to split PDFs but is not installed in this environment.")
        return ""

    try:
        _, blob_name = parse_gcs_uri(gcs_uri)
    except ValueError as exc:
        logger.error("Invalid GCS URI: %s", exc)
        return ""

    try:
        file_bytes = gcs_manager.download_blob(blob_name)
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(pdf_reader.pages)
        logger.info("Document has %s page(s).", total_pages)
    except Exception as exc:
        logger.error("Failed to download or read PDF from GCS %s: %s", blob_name, exc)
        return ""

    if total_pages == 0:
        logger.warning("Document %s contains no pages.", gcs_uri)
        return ""

    chunk_ranges: Sequence[ChunkRange] = calculate_page_chunks(total_pages)
    logger.info(
        "Splitting document into %s chunk(s) capped at %s pages each.",
        len(chunk_ranges),
        PAGE_LIMIT,
    )

    temp_blob_names: List[str] = []

    docai_client: documentai.DocumentProcessorServiceClient | None = None
    processor_resource: str | None = None

    try:
        client_options = {"api_endpoint": f"{location}-documentai.googleapis.com"}
        docai_client = documentai.DocumentProcessorServiceClient(client_options=client_options)
        processor_resource = docai_client.processor_path(project_id, location, processor_id)

        extracted_chunks: List[str] = []

        for start_page, end_page in chunk_ranges:
            pdf_writer = PdfWriter()
            for page_index in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_index])

            chunk_bytes_io = io.BytesIO()
            pdf_writer.write(chunk_bytes_io)
            chunk_bytes = chunk_bytes_io.getvalue()

            chunk_blob_name = (
                f"deals/{deal_id}/docai_chunks/{deal_id}_p{start_page + 1}-p{end_page}.pdf"
            )
            chunk_uri = gcs_manager.upload_blob_from_bytes(
                data=chunk_bytes,
                destination_blob_name=chunk_blob_name,
            )
            temp_blob_names.append(chunk_blob_name)
            logger.info(
                "Uploaded chunk %s covering pages %s-%s (%s page(s)).",
                chunk_blob_name,
                start_page + 1,
                end_page,
                end_page - start_page,
            )

            try:
                text_chunk = extract_text_from_pdf_docai(
                    gcs_uri=chunk_uri,
                    project_id=project_id,
                    location=location,
                    processor_id=processor_id,
                    client=docai_client,
                    processor_resource=processor_resource,
                )
            except DocumentAIProcessingError as exc:
                logger.error("Failed to process chunk %s: %s", chunk_uri, exc)
                return ""

            if text_chunk:
                extracted_chunks.append(text_chunk)
            else:
                logger.warning("Chunk %s returned no text after processing.", chunk_uri)

        full_text = "\n\n".join(extracted_chunks)
        logger.info("All chunks processed and combined (total %s characters).", len(full_text))
        return full_text
    finally:  # pragma: no cover - cleanup requires integration environment
        logger.info("Cleaning up %s temporary chunk(s)...", len(temp_blob_names))
        for blob_name in temp_blob_names:
            gcs_manager.delete_blob(blob_name)

        if docai_client is not None:
            try:
                docai_client.transport.close()
            except AttributeError:
                pass
