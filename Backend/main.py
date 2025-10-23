from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from google.cloud import storage

from app.api.risk import router as risk_router
from config.settings import settings
from models.schemas import (
    DealMetadata,
    MemoResponse,
    ProcessingStatus,
    Weightage,
    ChatRequest,
    ChatResponse,
)
from utils.cache_utils import build_weight_signature
from utils.docx_utils import MemoExporter
from utils.firestore_utils import FirestoreManager
from utils.gcs_utils import GCSManager
from utils.summarizer import GeminiSummarizer
from utils.chat_agent import StartupChatAgent

from dotenv import load_dotenv
load_dotenv()

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- FastAPI app (for Jupyter proxy support) ----------
PORT = os.getenv("PORT", "9000")
ROOT_PATH = f"/proxy/{PORT}"

app = FastAPI(
    title="AI Investment Memo Generator",
    description="Generate investor-ready memos from pitch materials",
    version="1.0.0",
    root_path=ROOT_PATH,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(risk_router)

# ---------- Initialize services ----------
gcs_manager = GCSManager()
gemini_summarizer = GeminiSummarizer()
memo_exporter = MemoExporter()
firestore_manager = FirestoreManager()
chat_agent = StartupChatAgent()

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"status": "ok", "service": "investment-memo", "docs": f"{ROOT_PATH}/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/upload", response_model=dict)
async def upload_deal(
    pitch_deck: UploadFile = File(...),
):
    """Upload deal materials and start processing"""
    try:
        print("upload_deal called")
        # Generate unique deal ID
        # deal_id = f"{company_name.lower().replace(' ', '')}_{uuid.uuid4().hex[:6]}"
        deal_id = f"{uuid.uuid4().hex[:6]}"

        # Create deal metadata
        metadata = DealMetadata(
            deal_id=deal_id,
            status="queued",
            created_at=datetime.utcnow(),
        )

        # Save initial metadata to Firestore
        await firestore_manager.create_deal(deal_id, metadata.dict())

        # Upload pitch deck to GCS
        file_urls: Dict[str, Any] = {}
        deck_hash: Optional[str] = None
        if pitch_deck:
            pitch_deck_url, deck_hash = await gcs_manager.upload_file(
                pitch_deck, f"deals/{deal_id}/pitch_deck.pdf"
            )
            file_urls['pitch_deck_url'] = pitch_deck_url

        # Update Firestore with file URLs and hash metadata if available
        updates: Dict[str, Any] = {"raw_files": file_urls}
        if deck_hash:
            updates["metadata.deck_hash"] = deck_hash
        await firestore_manager.update_deal(deal_id, updates)

        return {
            "deal_id": deal_id,
            # "company_name": company_name,
            "status": "queued",
            "files": file_urls,
            "message": "Files uploaded successfully. Automated processing will begin shortly."
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{deal_id}", response_model=ProcessingStatus)
async def get_processing_status(deal_id: str):
    """Get current processing status"""
    try:
        deal_data = await firestore_manager.get_deal(deal_id)
        if not deal_data:
            raise HTTPException(status_code=404, detail="Deal not found")

        return ProcessingStatus(**deal_data.get('metadata', {}))

    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_memo/{deal_id}", response_model=MemoResponse)
async def generate_memo(deal_id: str, weightage: Weightage = Body(...)):
    """Generate investment memo"""
    try:
        deal_data = await firestore_manager.get_deal(deal_id)
        if not deal_data:
            raise HTTPException(status_code=404, detail="Deal not found")

        if deal_data.get('metadata', {}).get('status') != 'processed':
            raise HTTPException(status_code=400, detail="Deal processing not complete")

        metadata = deal_data.get('metadata', {})
        deck_hash = metadata.get('deck_hash')
        weight_dict = weightage.dict()
        await firestore_manager.update_deal(deal_id, {
            "metadata.weightage": weight_dict
        })

        weight_signature = build_weight_signature(weight_dict)
        cached_memo_entry = await firestore_manager.get_cached_memo(deck_hash, weight_signature)

        memo_text = None
        if cached_memo_entry:
            memo_text = cached_memo_entry.get("memo_json") or cached_memo_entry.get("memo_text")

        if memo_text is None:
            memo_text = await gemini_summarizer.generate_memo(deal_data, weight_dict)
            if deck_hash:
                await firestore_manager.cache_memo(deck_hash, weight_signature, memo_text, weight_dict)
            from_cache = False
        else:
            from_cache = True

        docx_url = await memo_exporter.create_memo_docx(deal_id, memo_text)

        memo_data = {
            "draft_v1": memo_text,
            "docx_url": docx_url,
            "generated_at": datetime.utcnow()
        }
        if from_cache:
            memo_data["cached_from_deck"] = True

        await firestore_manager.update_deal(deal_id, {"memo": memo_data, "metadata.memo_cached_from_hash": from_cache})

        return MemoResponse(
            deal_id=deal_id,
            memo_text=memo_text,
            docx_url=docx_url,
            all_data=deal_data
        )

    except Exception as e:
        logger.error(f"Memo generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/interview", response_model=ChatResponse)
async def interview_chat(request: ChatRequest) -> ChatResponse:
    """Respond to chatbot interactions using memo context."""

    try:
        history_payload = [message.model_dump() for message in request.history]
        reply = await chat_agent.generate_response(request.analysis_data, history_payload)
        return ChatResponse(message=reply)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/service failures
        logger.error("Chat generation error: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to generate chat response") from exc


    except Exception as e:
        logger.error(f"Processing error for deal {deal_id}: {str(e)}")
        await firestore_manager.update_deal(deal_id, {
            "metadata.status": "error",
            "metadata.error": str(e)
        })

@app.get("/deals", response_model=list)
async def fetch_all_deals():
    """Fetch all deals from Firestore"""
    try:
        # all_deals = await firestore_manager.get_all_deals()  # You need to implement this in FirestoreManager
        all_deals = await firestore_manager.list_deals()  # You need to implement this in FirestoreManager
    
        return all_deals
    except Exception as e:
        logger.error(f"Fetch all deals error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deals/{deal_id}", response_model=dict)
async def fetch_specific_deal(deal_id: str):
    """Fetch a specific deal by deal_id"""
    try:
        deal = await firestore_manager.get_deal(deal_id)
        if not deal:
            raise HTTPException(status_code=404, detail="Deal not found")
        return deal
    except Exception as e:
        logger.error(f"Fetch deal error for {deal_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/deals/{deal_id}", response_model=dict)
async def delete_specific_deal(deal_id: str):
    """Delete a specific deal by deal_id"""
    try:
        deal = await firestore_manager.get_deal(deal_id)
        if not deal:
            raise HTTPException(status_code=404, detail="Deal not found")

        await firestore_manager.delete_deal(deal_id)  # You need to implement this in FirestoreManager

        return {"deal_id": deal_id, "status": "deleted", "message": "Deal deleted successfully"}
    except Exception as e:
        logger.error(f"Delete deal error for {deal_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/download_memo/{deal_id}")
async def download_memo(deal_id: str):
    deal_data = await firestore_manager.get_deal(deal_id)
    print(deal_data)
    if not deal_data or "memo" not in deal_data:
        raise HTTPException(status_code=404, detail="Memo not found")

    gs_url = deal_data["memo"]["docx_url"]

    # Download the file from GCS to a temporary local file
    local_path = f"/tmp/{deal_id}_memo.docx"
    await gcs_manager.download_file(gs_url, local_path)  # implement download_file in GCSManager

    # Return as a downloadable file
    return StreamingResponse(
        open(local_path, "rb"),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={deal_id}_memo.docx"}
    )

@app.get("/download_pitch_deck/{deal_id}")
async def download_pitch_deck(deal_id: str):
    """
    Download the pitch deck for a deal from GCS.
    """
    try:
        # Fetch deal from Firestore
        deal = await firestore_manager.get_deal(deal_id)
        if not deal:
            raise HTTPException(status_code=404, detail="Deal not found")

        # Get pitch deck URL
        gcs_path = deal.get("raw_files", {}).get("pitch_deck_url")
        if not gcs_path:
            raise HTTPException(status_code=404, detail="Pitch deck not found")

        # Parse bucket and blob name from gs:// URL
        assert gcs_path.startswith("gs://")
        parts = gcs_path[5:].split("/", 1)
        bucket_name, blob_name = parts[0], parts[1]

        # Download file from GCS into memory
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        file_stream = io.BytesIO()
        blob.download_to_file(file_stream)
        file_stream.seek(0)

        filename = blob_name.split("/")[-1]
        return StreamingResponse(
            file_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Download pitch deck error for deal {deal_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
