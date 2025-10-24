import io
import logging
from urllib.parse import urlparse
from pypdf import PdfReader, PdfWriter

from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import ProcessOptions

# Import our singleton GCSManager instance
from .gcs_utils import gcs_manager 

logger = logging.getLogger(__name__)

PAGE_LIMIT = 15  # The hard quota for standard Document AI OCR

def parse_gcs_uri(gcs_uri: str) -> (str, str):
    """Parses a GCS URI into bucket and blob name."""
    parsed_uri = urlparse(gcs_uri)
    if parsed_uri.scheme != "gs":
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name = parsed_uri.netloc
    blob_name = parsed_uri.path.lstrip('/')
    return bucket_name, blob_name

def extract_text_from_pdf_docai(
    gcs_uri: str,
    project_id: str,
    location: str,
    processor_id: str,
) -> str:
    """
    Processes a SINGLE document chunk (<= 15 pages) using Document AI.
    """
    logger.info(f"Starting Document AI processing for chunk: {gcs_uri}")
    client_options = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    name = client.processor_path(project_id, location, processor_id)

    gcs_document = documentai.GcsDocument(
        gcs_uri=gcs_uri, mime_type="application/pdf"
    )

    # We are NOT using imageless mode, so we can analyze images.
    request = documentai.ProcessRequest(
        name=name,
        gcs_document=gcs_document,
        skip_human_review=True
    )

    try:
        result = client.process_document(request=request)
        document = result.document
        logger.info(f"Document AI processing complete for chunk: {gcs_uri}")
        return document.text
    except Exception as e:
        logger.error(f"Error in Document AI processing chunk {gcs_uri}: {e}")
        return ""

def process_large_pdf(
    gcs_uri: str,
    deal_id: str,
    project_id: str,
    location: str,
    processor_id: str,
    bucket_name: str
) -> str:
    """
    Orchestrator to process large PDFs by splitting them into chunks
    that respect the Document AI 15-page limit.
    """
    logger.info(f"Starting large PDF processing for {gcs_uri}")

    try:
        _, blob_name = parse_gcs_uri(gcs_uri)
    except ValueError as e:
        logger.error(f"Invalid GCS URI: {e}")
        return ""

    # 1. Download the original PDF from GCS into memory
    try:
        file_bytes = gcs_manager.download_blob(blob_name)
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(pdf_reader.pages)
        logger.info(f"Document has {total_pages} pages. Splitting into chunks of {PAGE_LIMIT}.")
    except Exception as e:
        logger.error(f"Failed to download or read PDF from GCS {blob_name}: {e}")
        return ""

    if total_pages == 0:
        return ""

    # Optimization: If doc is small, process it directly
    if total_pages <= PAGE_LIMIT:
        logger.info("Document is under page limit. Processing directly.")
        return extract_text_from_pdf_docai(
            gcs_uri=gcs_uri,
            project_id=project_id,
            location=location,
            processor_id=processor_id
        )

    all_extracted_text = []
    chunk_gcs_uris = []
    temp_blob_names = []

    try:
        # 2. Split PDF into chunks and upload them
        for start_page in range(0, total_pages, PAGE_LIMIT):
            end_page = min(start_page + PAGE_LIMIT, total_pages)
            pdf_writer = PdfWriter()

            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            chunk_bytes_io = io.BytesIO()
            pdf_writer.write(chunk_bytes_io)
            chunk_bytes_io.seek(0)
            chunk_bytes = chunk_bytes_io.getvalue()

            chunk_file_name = f"deals/{deal_id}/temp_chunk_p{start_page + 1}-p{end_page}.pdf"

            gcs_manager.upload_blob_from_bytes(
                data=chunk_bytes,
                destination_blob_name=chunk_file_name
            )

            chunk_gcs_uri = f"gs://{bucket_name}/{chunk_file_name}"
            chunk_gcs_uris.append(chunk_gcs_uri)
            temp_blob_names.append(chunk_file_name)
            logger.info(f"Uploaded chunk {chunk_gcs_uri} ({end_page - start_page} pages)")

        # 3. Process each chunk using the *existing* DocAI function
        logger.info("Processing all chunks with Document AI...")
        for chunk_uri in chunk_gcs_uris:
            text_chunk = extract_text_from_pdf_docai(
                gcs_uri=chunk_uri,
                project_id=project_id,
                location=location,
                processor_id=processor_id
            )
            if text_chunk:
                all_extracted_text.append(text_chunk)
            else:
                logger.warning(f"Warning: Chunk {chunk_uri} returned no text.")

        # 4. Combine all the text
        full_text = "\n\n".join(all_extracted_text)
        logger.info("All chunks processed and combined.")
        return full_text

    finally:
        # 5. Clean up the temporary chunk files
        logger.info(f"Cleaning up {len(temp_blob_names)} temporary chunks...")
        for blob_name in temp_blob_names:
            gcs_manager.delete_blob(blob_name)
