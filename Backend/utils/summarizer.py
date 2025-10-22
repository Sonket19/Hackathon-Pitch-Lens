from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import vertexai
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from config.settings import settings

logger = logging.getLogger(__name__)


def _ensure_vertex_initialised() -> None:
    """Initialise Vertex AI once per process."""

    try:
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    except ValueError:  # pragma: no cover - vertexai raises if already initialised
        logger.debug("Vertex AI initialisation may have already occurred", exc_info=True)


class GeminiSummarizer:
    """Wrapper around Gemini with deterministic defaults and parsing helpers."""

    def __init__(self) -> None:
        _ensure_vertex_initialised()
        self.model = GenerativeModel("gemini-2.5-pro")
        # Force deterministic behaviour so repeated uploads stay consistent.
        self._generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )

        # Multimodal audio/video model for transcripts and tonal analysis.
        self._media_model = GenerativeModel("gemini-1.5-pro")
        self._media_config = GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            top_k=32,
            max_output_tokens=4096,
        )

    def _generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config=self._generation_config,
        )
        text = getattr(response, "text", "")
        return text.strip() if isinstance(text, str) else ""

    def _generate_multimodal(
        self,
        contents: Sequence[Content | str | Part],
        *,
        generation_config: Optional[GenerationConfig] = None,
        model: Optional[GenerativeModel] = None,
    ):
        active_model = model or self.model
        config = generation_config or self._generation_config
        return active_model.generate_content(
            list(contents),
            generation_config=config,
        )

    def generate_text(self, prompt: str) -> str:
        """Public wrapper for deterministic text generation."""
        return self._generate_text(prompt)

    @staticmethod
    def _coerce_string_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [part.strip("-• \t") for part in value.splitlines() if part.strip("-• \t")]
            return GeminiSummarizer._coerce_string_list(parsed)
        return []

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _strip_json_fences(payload: str) -> str:
        return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", payload, flags=re.MULTILINE).strip()

    async def _legacy_summarize_pitch_deck(self, full_text: str) -> Dict[str, Any]:
        summary_prompt = (
            "Analyze the following pitch deck content and extract information for these sections:\n"
            "- problem: What problem is being solved?\n"
            "- solution: What is the proposed solution?\n"
            "- market: Market size, opportunity, and target customers\n"
            "- team: Information about the founding team and key personnel\n"
            "- traction: Current progress, metrics, customers, revenue\n"
            "- financials: Financial projections, funding requirements, revenue model\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return the analysis as a JSON object with the above keys. Be concise but comprehensive.\n"
            'If a section is not clearly addressed in the pitch deck, indicate "Not specified" for that key.'
        )

        summary_text = self._generate_text(summary_prompt)

        founder_prompt = (
            "Analyze the following pitch deck content and extract list of founders in array:\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return the analysis as an array.\n"
            "If no data found send empty array."
        )

        founder_raw = self._generate_text(founder_prompt)
        founder_clean = self._strip_json_fences(founder_raw)
        try:
            founder_data = json.loads(founder_clean) if founder_clean else []
        except json.JSONDecodeError:
            founder_data = founder_clean
        founder_response = self._dedupe_preserve_order(self._coerce_string_list(founder_data))

        sector_prompt = (
            "Analyze the following pitch deck content and extract name of sector in which this startup fall in:\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return specific sector name only no extra word.\n"
            'If no data found send empty string "".'
        )

        sector_response = self._strip_json_fences(self._generate_text(sector_prompt))

        company_name_prompt = (
            "Analyze the following pitch deck content and extract name of the startup/company this pitch is for:\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return the organization or company name only with no extra words.\n"
            'If no data found send empty string "".'
        )

        company_name_response = self._strip_json_fences(self._generate_text(company_name_prompt))

        product_name_prompt = (
            "Analyze the following pitch deck content and extract the primary product or platform name the startup is promoting.\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            'Return the product/solution name only with no extra words. If none is mentioned, return an empty string "".'
        )

        product_name_response = self._strip_json_fences(self._generate_text(product_name_prompt))

        return {
            "summary_res": summary_text,
            "founder_response": founder_response,
            "sector_response": sector_response,
            "company_name_response": company_name_response,
            "product_name_response": product_name_response,
        }

    async def summarize_pitch_deck(
        self,
        full_text: str,
        *,
        gcs_uri: Optional[str] = None,
        mime_type: str = "application/pdf",
    ) -> Dict[str, Any]:
        """Summarize pitch deck into structured sections."""
        try:
            prompt = (
                "You are processing a startup pitch deck. Review the attached deck as well as the OCR text to produce a structured JSON summary.\n"
                "Tasks:\n"
                "1. Extract the company name, primary product name, founders, and sector.\n"
                "2. Derive a concise narrative summary that references key traction and go-to-market details.\n"
                "3. Examine charts, tables, and financial slides to pull out ARR/MRR, growth, fundraising, or unit economics figures explicitly.\n"
                "4. Evaluate slide design, visual polish, and information architecture to infer the business maturity stage (mention this in the summary).\n"
                "5. Return valid JSON with keys: summary, founders (array), sector, company_name, product_name.\n"
                "   - Mention visual maturity observations in the summary string.\n"
                "   - If a value is unknown, use an empty string or empty array.\n"
            )

            contents: List[Content | str | Part] = []
            if gcs_uri:
                contents.append(Part.from_uri(gcs_uri, mime_type=mime_type))

            ocr_section = (
                "OCR_TEXT_START\n"
                f"{full_text}\n"
                "OCR_TEXT_END"
            )
            contents.append(prompt + "\n\n" + ocr_section)

            structured_response = await asyncio.to_thread(
                self._generate_multimodal,
                contents,
            )

            structured_raw = self._extract_text(structured_response)
            structured_clean = self._strip_json_fences(structured_raw)

            try:
                structured_payload: Dict[str, Any] = json.loads(structured_clean) if structured_clean else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse structured summary payload; falling back to legacy prompts")
                return await self._legacy_summarize_pitch_deck(full_text)

            summary_text = str(structured_payload.get("summary", "")).strip()
            founder_response = self._dedupe_preserve_order(
                self._coerce_string_list(structured_payload.get("founders", []))
            )
            sector_response = str(structured_payload.get("sector", "")).strip()
            company_name_response = str(structured_payload.get("company_name", "")).strip()
            product_name_response = str(structured_payload.get("product_name", "")).strip()

            if not summary_text:
                summary_text = self._generate_text(
                    "Provide a concise summary of the following pitch deck in under 160 words, mentioning visual maturity signals:\n"
                    f"{full_text}"
                )

            return {
                "summary_res": summary_text,
                "founder_response": founder_response,
                "sector_response": sector_response,
                "company_name_response": company_name_response,
                "product_name_response": product_name_response,
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Pitch deck summarization error: %s", exc)
            return {
                "problem": "Error in processing",
                "solution": "Error in processing",
                "market": "Error in processing",
                "team": "Error in processing",
                "traction": "Error in processing",
                "financials": "Error in processing",
                "founder_response": [],
                "sector_response": "",
                "company_name_response": "",
                "product_name_response": "",
                "summary_res": "",
            }

    async def summarize_audio_transcript(self, transcript: str) -> str:
        """Summarize audio transcript."""
        try:
            prompt = (
                "Summarize the following pitch transcript into key points:\n"
                "- Main value proposition\n"
                "- Key business metrics mentioned\n"
                "- Important insights about market or competition\n"
                "- Notable quotes from the founder\n\n"
                "Transcript:\n"
                f"{transcript}\n\n"
                "Provide a concise summary in bullet points."
            )

            return self._generate_text(prompt)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Audio summarization error: %s", exc)
            return "Error processing audio transcript"

    async def analyze_media_from_gcs(self, gcs_uri: str, *, mime_type: str) -> Dict[str, Any]:
        """Leverage Gemini 1.5 Pro to analyse raw audio or video directly from GCS."""

        def _call() -> Dict[str, Any]:
            prompt = (
                "You are analysing an investor pitch recording.\n"
                "Return a JSON object with the following keys:\n"
                "- transcript: succinct transcript capturing the narrative (<= 1200 words).\n"
                "- summary: concise prose summary tailored for investors.\n"
                "- tone_analysis: short description of speaker tone and pacing.\n"
                "- pacing_observations: any pacing or delivery notes.\n"
                "- notable_quotes: array of short impactful quotes if available.\n"
                "Ensure the JSON is valid and omit commentary outside the object."
            )

            response = self._generate_multimodal(
                [Part.from_uri(gcs_uri, mime_type=mime_type), prompt],
                generation_config=self._media_config,
                model=self._media_model,
            )
            raw_text = self._extract_text(response)
            cleaned = self._strip_json_fences(raw_text)
            return json.loads(cleaned) if cleaned else {}

        try:
            payload = await asyncio.to_thread(_call)
        except json.JSONDecodeError as exc:  # pragma: no cover - guard unexpected formats
            logger.error("Failed to decode media analysis payload: {0}".format(exc))
            payload = {}
        except Exception as exc:  # pragma: no cover - network/service error resilience
            logger.error("Media analysis error: %s", exc)
            payload = {}

        transcript = str(payload.get("transcript", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        tone = str(payload.get("tone_analysis", "")).strip()
        pacing = str(payload.get("pacing_observations", "")).strip()
        quotes = payload.get("notable_quotes") if isinstance(payload.get("notable_quotes"), list) else []

        return {
            "raw": transcript,
            "concise": {
                "summary": summary,
                "tone_analysis": tone,
                "pacing_observations": pacing,
                "notable_quotes": quotes,
            },
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", "")
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return text if isinstance(text, str) else ""

        chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)
        joined = "".join(chunks)
        return joined or (text if isinstance(text, str) else "")

    async def generate_memo(self, deal_data: Dict[str, Any], weightage: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete investment memo."""
        try:
            metadata = deal_data.get("metadata", {})
            extracted_text = deal_data.get("extracted_text", {})
            public_data = deal_data.get("public_data", {})
            user_input = deal_data.get("user_input", {})

            context = self._build_memo_context(metadata, extracted_text, public_data, user_input)

            prompt = f"""
                You are an investment analyst. Your task is to generate a structured investment memo for the startup under review.

                Instructions:
                1. The output MUST be in strict JSON format. Do not include text outside the JSON.
                2. Always follow the schema below exactly. Every key and subkey MUST appear, even if data is missing (use "Not available" or [] for empty).
                3. The response must be deterministic and stable across runs — always yielding the same result unless the input data itself changes.
                4. Use the provided company data first, then enrich with reliable internet sources about competitors, market size, and industry trends.
                5. For probabilistic forecasting of company claims, follow these rules:
                   - Dataset length 6–12 months → ETS or Bayesian log-linear regression
                   - Dataset length 12–18 months → ARIMA or Holt-Winters
                   - Dataset length 18+ months → Prophet or Bayesian Structural Time Series
                   - If historical time-series is unavailable, run Monte Carlo simulations on pipeline data
                6. Construct a composite risk metric by combining:
                   - Burn rate
                   - Runway
                   - Gross margins
                   - ARR growth
                   - CAC/LTV ratios
                   - Credibility of claims (based on probabilistic analysis)
                   The risk metric should output a numeric score (0–100) and a narrative justification.
                7. Keep analysis fact-based and consistent. Do not invent competitors or valuations without attribution.
                8. All financial projections, probabilities, and risk metrics must be deterministic and stable.

                Schema to follow exactly:
                {{
                  "company_overview": {{
                    "name": "string",
                    "sector": "string",
                    "founders": [
                      {{
                        "name": "string",
                        "education": "string",
                        "professional_background": "string",
                        "previous_ventures": "string"
                      }}
                    ],
                    "technology": "string"
                  }},
                  "market_analysis": {{
                    "industry_size_and_growth": {{
                      "total_addressable_market": {{
                        "name": "string",
                        "value": "string",
                        "cagr": "string",
                        "source": "string"
                      }},
                      "serviceable_obtainable_market": {{
                        "name": "string",
                        "value": "string",
                        "cagr": "string",
                        "source": "string"
                      }},
                      "commentary": "string"
                    }},
                    "recent_news": "string",
                    "competitor_details": [{{
                      "name": "string",
                      "business_model": "string",
                      "funding": "string",
                      "margins": "string",
                      "commentary": "string",
                      "category": "string"
                    }}],
                    "sub_segment_opportunities": ["string"]
                  }},
                  "business_model": {{
                    "revenue_streams": "string",
                    "pricing": "string",
                    "scalability": "string",
                    "unit_economics": {{
                      "customer_lifetime_value_ltv": "string",
                      "customer_acquisition_cost_cac": "string"
                    }}
                  }},
                  "financials": {{
                    "funding_history": "string",
                    "projections": [{{
                      "year": "string",
                      "revenue": "string"
                    }}],
                    "valuation_rationale": "string",
                    "srr_mrr": {{
                      "current_booked_arr": "string",
                      "current_mrr": "string"
                    }},
                    "burn_and_runway": {{
                      "funding_ask": "string",
                      "stated_runway": "string",
                      "implied_net_burn": "string"
                    }}
                  }},
                  "claims_analysis": [{{
                    "claim": "string",
                    "analysis_method": "string",
                    "input_dataset_length": "string",
                    "simulated_probability": "string",
                    "result": "string",
                    "simulation_assumptions": {{"assumptions": "string"}}
                  }}],
                  "risk_metrics": {{
                    "composite_risk_score": 0,
                    "score_interpretation": "string",
                    "narrative_justification": "string"
                  }},
                  "conclusion": {{
                    "overall_attractiveness": "string"
                  }}
                }}

                Weighting preferences:
                {weightage}

                Source information:
                {context}
            """

            raw_response = self._generate_text(prompt)
            clean = re.sub(r"^```json\s*|\s*```$", "", raw_response, flags=re.MULTILINE).strip()
            try:
                return json.loads(clean) if clean else {}
            except json.JSONDecodeError:
                logger.warning("Gemini memo response was not valid JSON; storing raw payload under 'raw_text'")
                return {"raw_text": clean}

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Memo generation error: %s", exc)
            return {"error": "Error generating memo"}

    def _build_memo_context(
        self,
        metadata: Dict[str, Any],
        extracted_text: Dict[str, Any],
        public_data: Dict[str, Any],
        user_input: Dict[str, Any],
    ) -> str:
        """Build context string for memo generation."""

        context_parts: List[str] = []

        # Company information
        context_parts.append(f"Company: {metadata.get('company_name', 'N/A')}")
        context_parts.append(f"Founders: {', '.join(metadata.get('founder_names', [])) or 'N/A'}")
        context_parts.append(f"Sector: {metadata.get('sector', 'N/A')}")

        # Pitch deck insights
        pitch_deck = extracted_text.get("pitch_deck") if isinstance(extracted_text, dict) else None
        if isinstance(pitch_deck, dict):
            concise = pitch_deck.get("concise")
            if concise:
                context_parts.append("Pitch Deck Analysis:")
                context_parts.append(str(concise))

        # Media summaries
        for media_type in ("voice_pitch", "video_pitch"):
            media_data = extracted_text.get(media_type) if isinstance(extracted_text, dict) else None
            if isinstance(media_data, dict):
                summary = media_data.get("concise", {}).get("summary")
                if summary:
                    context_parts.append(f"{media_type.replace('_', ' ').title()} Summary:")
                    context_parts.append(str(summary))

        # Public data
        if isinstance(public_data, dict) and public_data:
            context_parts.append("Public Information:")
            for key, value in public_data.items():
                if isinstance(value, list):
                    context_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
                else:
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        # User input (Q&A / weightages)
        if isinstance(user_input, dict) and user_input:
            qna = user_input.get("qna")
            if isinstance(qna, dict):
                context_parts.append("Additional Q&A:")
                for question, answer in qna.items():
                    context_parts.append(f"Q: {question}")
                    context_parts.append(f"A: {answer}")

            if "weightages" in user_input:
                context_parts.append(f"Evaluation Weightages: {user_input['weightages']}")

        return "\n".join(context_parts)
