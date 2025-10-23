"""Async orchestration of deal processing triggered by Cloud Functions."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

from utils.firestore_utils import FirestoreManager
from utils.ocr_utils import PDFProcessor
from utils.search_utils import PublicDataGatherer
from utils.email_utils import extract_emails
from utils.naming import build_company_display_name


logger = logging.getLogger(__name__)


class DealProcessingOrchestrator:
    """Encapsulates the full ingestion pipeline for a deal upload."""

    def __init__(self) -> None:
        self.firestore = FirestoreManager()
        self.pdf_processor = PDFProcessor()
        self.data_gatherer = PublicDataGatherer()

    def process(self, deal_id: str) -> None:
        """Entry point for synchronous callers (e.g., Cloud Functions)."""

        asyncio.run(self._process_async(deal_id))

    async def _process_async(self, deal_id: str) -> None:
        await self.firestore.update_deal(deal_id, {"metadata.status": "processing"})

        deal_snapshot = await self.firestore.get_deal(deal_id) or {}
        raw_files = deal_snapshot.get("raw_files", {}) or {}
        metadata_snapshot = deal_snapshot.get("metadata", {}) or {}
        deck_hash = metadata_snapshot.get("deck_hash")

        extracted_text: Dict[str, Any] = {}
        temp_res: Dict[str, Any] = {}
        public_data: Dict[str, Any] = {}
        stage_timings: Dict[str, Any] = {}

        cache_bundle = await self.firestore.get_cached_deck(deck_hash)
        cache_hit = bool(cache_bundle and cache_bundle.get('summary'))
        if cache_hit:
            logger.info("Reusing cached analysis for deal %s (hash %s)", deal_id, deck_hash)
            temp_res = cache_bundle.get('summary', {}) or {}
            extracted_text = cache_bundle.get('extracted_text', {}) or {}
            public_data = cache_bundle.get('public_data', {}) or {}
            stage_timings['cache_hit'] = True

            if 'pitch_deck' not in extracted_text:
                logger.info("Cached payload missing raw pitch deck for deal %s; reprocessing", deal_id)
                cache_hit = False
                temp_res = {}
                extracted_text = {}
                public_data = {}

        logos_detected: List[str] = []
        pitch_deck_uri = raw_files.get("pitch_deck_url") if isinstance(raw_files, dict) else None

        if not cache_hit and isinstance(pitch_deck_uri, str) and pitch_deck_uri:
            logger.info("Processing PDF for deal %s", deal_id)
            pdf_start = time.perf_counter()
            pdf_data = await self.pdf_processor.process_pdf(pitch_deck_uri)
            stage_timings['pdf_processing_s'] = time.perf_counter() - pdf_start

            summary_snapshot = {
                "concise": pdf_data.get("concise", ""),
                "founder_response": pdf_data.get("founder_response", []),
                "sector_response": pdf_data.get("sector_response", ""),
                "company_name_response": pdf_data.get("company_name_response", ""),
                "product_name_response": pdf_data.get("product_name_response", ""),
            }
            temp_res = summary_snapshot
            extracted_text = {
                "pitch_deck": {
                    "raw": pdf_data.get("raw", {}),
                    "concise": summary_snapshot["concise"],
                    "logos": pdf_data.get("logos", []),
                }
            }
            logos_detected = pdf_data.get("logos", []) or []

        if not logos_detected:
            pitch_payload = extracted_text.get("pitch_deck", {}) if isinstance(extracted_text, dict) else {}
            if isinstance(pitch_payload, dict):
                logos_detected = pitch_payload.get("logos", []) or []

        if not temp_res:
            raise ValueError("Pitch deck summary could not be generated")

        company_name = temp_res.get("company_name_response", "")
        product_name = temp_res.get("product_name_response", "")
        company_for_search = company_name or metadata_snapshot.get('company_legal_name', "")

        raw_founders = temp_res.get("founder_response", []) or []
        if isinstance(raw_founders, list):
            founders_for_search = [str(name).strip() for name in raw_founders if str(name).strip()]
        elif isinstance(raw_founders, str) and raw_founders.strip():
            founders_for_search = [raw_founders.strip()]
        else:
            founders_for_search = []

        sector_for_search = temp_res.get("sector_response", "")

        if not public_data:
            logger.info("Gathering public data for deal %s", deal_id)
            public_start = time.perf_counter()
            public_data = await self.data_gatherer.gather_data(
                company_for_search,
                founders_for_search,
                sector_for_search,
                logos=logos_detected,
            )
            stage_timings['public_data_s'] = time.perf_counter() - public_start

        if deck_hash and not cache_hit:
            await self.firestore.set_cached_deck(
                deck_hash,
                {
                    "summary": temp_res,
                    "extracted_text": extracted_text,
                    "public_data": public_data,
                },
            )

        display_name = build_company_display_name(company_name, product_name)
        founders = list(dict.fromkeys(founders_for_search))

        existing_email_values: List[str] = []
        snapshot_emails = metadata_snapshot.get('founder_emails')
        if isinstance(snapshot_emails, list):
            existing_email_values.extend(str(item) for item in snapshot_emails if isinstance(item, str))
        snapshot_contact = metadata_snapshot.get('contact_email')
        if isinstance(snapshot_contact, str) and snapshot_contact.strip():
            existing_email_values.append(snapshot_contact.strip())

        founder_emails = extract_emails(
            existing_email_values,
            extracted_text,
            public_data,
            deal_snapshot.get('memo', {}),
        )

        contact_email = snapshot_contact.strip() if isinstance(snapshot_contact, str) and snapshot_contact.strip() else None
        if founder_emails and not contact_email:
            contact_email = founder_emails[0]

        logo_companies = public_data.get("logo_companies", []) if isinstance(public_data, dict) else []

        update_payload: Dict[str, Any] = {
            "extracted_text": extracted_text,
            "public_data": public_data,
            "metadata.status": "processed",
            "metadata.processed_at": datetime.utcnow(),
            "metadata.company_name": company_name or product_name,
            "metadata.display_name": display_name,
            "metadata.company_legal_name": company_name,
            "metadata.product_name": product_name,
            "metadata.names": {
                "company": company_name,
                "product": product_name,
                "display": display_name,
            },
            "metadata.founder_names": founders,
            "metadata.sector": sector_for_search,
            "metadata.cached_from_hash": cache_hit,
        }

        if founder_emails:
            update_payload["metadata.founder_emails"] = founder_emails
        if contact_email:
            update_payload["metadata.contact_email"] = contact_email
        if logo_companies:
            update_payload["metadata.logo_companies"] = logo_companies

        write_start = time.perf_counter()
        await self.firestore.update_deal(
            deal_id,
            update_payload,
        )
        stage_timings['firestore_write_s'] = time.perf_counter() - write_start

        if stage_timings:
            timing_payload = {}
            for key, value in stage_timings.items():
                if isinstance(value, bool):
                    timing_payload[key] = value
                elif isinstance(value, (int, float)):
                    timing_payload[key] = round(float(value), 3)
                else:
                    timing_payload[key] = value

            logger.info(
                "Deal %s processing timings (s): %s",
                deal_id,
                timing_payload,
            )

        logger.info("Deal %s processed successfully", deal_id)
