"""Vector search index helpers for comparable deal retrieval."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import vertexai
from google.cloud import bigquery
from vertexai import language_models

from config.settings import settings


logger = logging.getLogger(__name__)


class DealVectorIndex:
    """Manage BigQuery-based vector embeddings for analysed deals."""

    def __init__(self) -> None:
        if not settings.BIGQUERY_DATASET or not settings.BIGQUERY_VECTOR_TABLE:
            raise ValueError("Vector index requires BIGQUERY_DATASET and BIGQUERY_VECTOR_TABLE environment variables")

        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
        self._embed_model = language_models.TextEmbeddingModel.from_pretrained(settings.VECTOR_EMBED_MODEL)
        self._bq = bigquery.Client(project=settings.GCP_PROJECT_ID)
        dataset = settings.BIGQUERY_DATASET
        if "." in dataset:
            dataset_ref = dataset
        else:
            dataset_ref = f"{settings.GCP_PROJECT_ID}.{dataset}"
        table = settings.BIGQUERY_VECTOR_TABLE
        if "." in table:
            self._table_id = table
        else:
            self._table_id = f"{dataset_ref}.{table}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert_deal(self, deal_id: str, memo: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Insert or update the deal representation in the vector index."""

        if not deal_id:
            logger.debug("Skipping vector index upsert because deal_id is missing")
            return

        summary = self._build_summary_payload(memo, metadata)
        embedding = self._embed_text(summary["summary_text"])

        query = f"""
            MERGE `{self._table_id}` T
            USING (
                SELECT
                    @deal_id AS deal_id,
                    @company_name AS company_name,
                    @sector AS sector,
                    @summary AS memo_summary,
                    @embedding AS embedding,
                    @timestamp AS updated_at
            ) S
            ON T.deal_id = S.deal_id
            WHEN MATCHED THEN
              UPDATE SET
                company_name = S.company_name,
                sector = S.sector,
                memo_summary = S.memo_summary,
                embedding = S.embedding,
                updated_at = S.updated_at
            WHEN NOT MATCHED THEN
              INSERT (deal_id, company_name, sector, memo_summary, embedding, updated_at)
              VALUES (S.deal_id, S.company_name, S.sector, S.memo_summary, S.embedding, S.updated_at)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("deal_id", "STRING", deal_id),
                bigquery.ScalarQueryParameter("company_name", "STRING", summary["company_name"]),
                bigquery.ScalarQueryParameter("sector", "STRING", summary["sector"]),
                bigquery.ScalarQueryParameter("summary", "STRING", summary["summary_text"]),
                bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding),
                bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", datetime.utcnow()),
            ]
        )

        job = self._bq.query(query, job_config=job_config)
        job.result()

    def find_similar_deals(
        self,
        memo: Dict[str, Any],
        metadata: Dict[str, Any],
        *,
        limit: int = 5,
        exclude_deal_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top K most similar deals based on cosine similarity."""

        summary = self._build_summary_payload(memo, metadata)
        embedding = self._embed_text(summary["summary_text"])

        query = f"""
            WITH target AS (
              SELECT @embedding AS embedding
            )
            SELECT
              deal_id,
              company_name,
              sector,
              memo_summary,
              1 - ML.DISTANCE(embedding, target.embedding, 'COSINE') AS similarity
            FROM `{self._table_id}`, target
            WHERE (@sector IS NULL OR sector = @sector)
              AND (@exclude IS NULL OR deal_id != @exclude)
            ORDER BY similarity DESC
            LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding),
                bigquery.ScalarQueryParameter("sector", "STRING", summary["sector"] or None),
                bigquery.ScalarQueryParameter("exclude", "STRING", exclude_deal_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        rows = self._bq.query(query, job_config=job_config).result()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> List[float]:
        embedding = self._embed_model.get_embeddings([text])[0]
        return list(getattr(embedding, "values", []))

    def _build_summary_payload(self, memo: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, str]:
        company = self._safe_get(memo, ["company_overview", "name"]) or metadata.get("company_name") or metadata.get("display_name") or "Unknown"
        sector = self._safe_get(memo, ["company_overview", "sector"]) or metadata.get("sector") or ""

        bullets: List[str] = [f"Company: {company}", f"Sector: {sector or 'Unknown'}"]

        founders = memo.get("company_overview", {}).get("founders", [])
        if isinstance(founders, list) and founders:
            bullets.append("Founders: " + ", ".join(str(item.get("name", "")) for item in founders if isinstance(item, dict)))

        revenue = self._safe_get(memo, ["financials", "srr_mrr", "current_booked_arr"])
        if revenue:
            bullets.append(f"Booked ARR: {revenue}")

        runway = self._safe_get(memo, ["financials", "burn_and_runway", "stated_runway"])
        if runway:
            bullets.append(f"Runway: {runway}")

        valuation = self._safe_get(memo, ["financials", "valuation_rationale"])
        if valuation:
            bullets.append(f"Valuation rationale: {valuation}")

        growth = self._safe_get(memo, ["market_analysis", "industry_size_and_growth", "commentary"])
        if growth:
            bullets.append(f"Market commentary: {growth}")

        claims = memo.get("claims_analysis", [])
        if isinstance(claims, list) and claims:
            top_claims = []
            for claim in claims[:3]:
                if isinstance(claim, dict):
                    statement = claim.get("claim")
                    probability = claim.get("simulated_probability")
                    if statement:
                        top_claims.append(f"{statement} (p={probability})" if probability else statement)
            if top_claims:
                bullets.append("Claims: " + "; ".join(top_claims))

        summary_text = " | ".join(part for part in bullets if part)

        return {
            "company_name": company,
            "sector": sector,
            "summary_text": summary_text,
        }

    @staticmethod
    def _safe_get(payload: Dict[str, Any], path: Sequence[str]) -> str:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                return ""
            current = current.get(key)
        if isinstance(current, str):
            return current
        if isinstance(current, (int, float)):
            return str(current)
        return ""

