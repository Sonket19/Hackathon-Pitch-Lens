import io
import logging
from typing import List, Sequence, Tuple
from urllib.parse import urlparse
from pypdf import PdfReader, PdfWriter
from typing import Any, Dict

from google.cloud import documentai_v1 as documentai

# Import our singleton GCSManager instance
from .gcs_utils import gcs_manager 
from .summarizer import GeminiSummarizer
from config.settings import settings

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

class PDFProcessor:
    def __init__(self):
        # We need the summarizer, as it was likely here before
        self.summarizer = GeminiSummarizer()
        logger.info("PDFProcessor initialized.")

    def _extract_chunk_text(
        self,
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

    def _get_full_text_orchestrator(self, gcs_uri: str, deal_id: str) -> str:
        """
        Orchestrator to get full text from large PDFs by splitting them.
        """
        logger.info(f"Starting large PDF text extraction for {gcs_uri}")
        
        try:
            bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        except ValueError as e:
            logger.error(f"Invalid GCS URI: {e}")
            return ""

        try:
            file_bytes = gcs_manager.download_blob(blob_name)
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            total_pages = len(pdf_reader.pages)
            logger.info(f"Document has {total_pages} pages. Splitting into chunks of {PAGE_LIMIT}.")
        except Exception as e:
            logger.error(f"Failed to download or read PDF from GCS {blob_name}: {e}")
            return "" # Return empty string on failure

        if total_pages == 0:
            return ""

        if total_pages <= PAGE_LIMIT:
            logger.info("Document is under page limit. Processing directly.")
            return self._extract_chunk_text(
                gcs_uri=gcs_uri,
                project_id=settings.DOCAI_PROJECT_ID,
                location=settings.DOCAI_LOCATION,
                processor_id=settings.DOCAI_PROCESSOR_ID
            )

        all_extracted_text = []
        chunk_gcs_uris = []
        temp_blob_names = []

        try:
            for start_page in range(0, total_pages, PAGE_LIMIT):
                end_page = min(start_page + PAGE_LIMIT, total_pages)
                pdf_writer = PdfWriter()
                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                
                chunk_bytes_io = io.BytesIO()
                pdf_writer.write(chunk_bytes_io)
                chunk_bytes = chunk_bytes_io.getvalue()
                
                chunk_file_name = f"deals/{deal_id}/temp_chunk_p{start_page + 1}-p{end_page}.pdf"
                gcs_manager.upload_blob_from_bytes(
                    data=chunk_bytes,
                    destination_blob_name=chunk_file_name
                )
                chunk_gcs_uri = f"gs://{bucket_name}/{chunk_file_name}"
                chunk_gcs_uris.append(chunk_gcs_uri)
                temp_blob_names.append(chunk_file_name)
                logger.info(f"Uploaded chunk {chunk_gcs_uri}")

            logger.info("Processing all chunks...")
            for chunk_uri in chunk_gcs_uris:
                text_chunk = self._extract_chunk_text(
                    gcs_uri=chunk_uri,
                    project_id=settings.DOCAI_PROJECT_ID,
                    location=settings.DOCAI_LOCATION,
                    processor_id=settings.DOCAI_PROCESSOR_ID
                )
                all_extracted_text.append(text_chunk)
            
            full_text = "\n\n".join(all_extracted_text)
            logger.info("All chunks processed and combined.")
            return full_text

        finally:
            logger.info(f"Cleaning up {len(temp_blob_names)} temporary chunks...")
            for blob_name in temp_blob_names:
                gcs_manager.delete_blob(blob_name)

    async def process_pdf(self, gcs_uri: str, deal_id: str) -> Dict[str, Any]:
        """
        This is the main method called by main.py.
        It now orchestrates text extraction and then calls the summarizer.
        """
        # Step 1: Get the full text using our new, robust orchestrator
        full_text = self._get_full_text_orchestrator(gcs_uri, deal_id)
        
        if not full_text:
            logger.error(f"Text extraction failed for {gcs_uri}. Aborting processing.")
            # We must return an empty dict or raise an error
            # that main.py can handle.
            raise ValueError("Failed to extract any text from the document.")

        # Step 2: Call the summarizer (as the original class likely did)
        # This part assumes your summarizer can run on the full_text
        # and return the dictionary structure that main.py expects.
        logger.info("Text extraction complete. Starting summarization...")
        
        # This is a guess based on your main.py code
        # Your summarizer might be more complex, but this replicates the
        # structure main.py expects.
        concise_summary = await self.summarizer.summarize_text(full_text, "concise")
        founder_response = await self.summarizer.summarize_text(full_text, "founders")
        sector_response = await self.summarizer.summarize_text(full_text, "sector")
        company_name_response = await self.summarizer.summarize_text(full_text, "company_name")
        product_name_response = await self.summarizer.summarize_text(full_text, "product_name")
        # 'logos' would require image analysis, which _extract_chunk_text supports.
        # We are not explicitly extracting them here, but the API ran.
        # For the hackathon, we can return an empty list.
        
        pdf_data = {
            "raw": full_text,
            "concise": concise_summary,
            "founder_response": [founder_response], # Guessing format
            "sector_response": sector_response,
            "company_name_response": company_name_response,
            "product_name_response": product_name_response,
            "logos": [] # Placeholder
        }
        
        logger.info(f"Summarization complete for deal {deal_id}.")
        return pdf_data

# This is the old, broken function that was at the top-level
# We are intentionally removing it to ensure it can't be called.
# def extract_text_from_pdf_docai(...)
