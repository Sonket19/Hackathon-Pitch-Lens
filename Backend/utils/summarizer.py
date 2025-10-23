from __future__ import annotations

import json
import logging
import re
import base64
import binascii
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, Part

from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiSummarizer:
    """Wrapper around Gemini with deterministic defaults and parsing helpers."""

    def __init__(self) -> None:
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
        self.model = GenerativeModel("gemini-2.5-pro")
        # Force deterministic behaviour so repeated uploads stay consistent.
        self._generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )

    def _generate_text(self, prompt: str, media_parts: Optional[List[Part]] = None) -> str:
        if media_parts:
            try:
                prompt_part: Union[str, Part] = Part.from_text(prompt)  # type: ignore[attr-defined]
                content: List[Union[str, Part]] = [prompt_part, *media_parts]
            except AttributeError:
                content = [prompt, *media_parts]
        else:
            content = prompt

        response = self.model.generate_content(
            content,
            generation_config=self._generation_config,
        )
        text = getattr(response, "text", "")
        return text.strip() if isinstance(text, str) else ""

    def generate_text(
        self,
        prompt: str,
        media_inputs: Optional[Sequence[Union[str, Tuple[Any, str], Dict[str, Any], bytes, bytearray]]] = None,
    ) -> str:
        """Public wrapper for deterministic text generation."""

        media_parts = self._prepare_media_parts(media_inputs)
        return self._generate_text(prompt, media_parts=media_parts)

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

    @staticmethod
    def _infer_mime_type(resource: str) -> Optional[str]:
        if not resource:
            return None
        mime, _ = mimetypes.guess_type(resource)
        if mime:
            return mime
        extension = Path(resource).suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(extension)

    def _prepare_media_parts(
        self,
        media_inputs: Optional[Sequence[Union[str, Tuple[Any, str], Dict[str, Any], bytes, bytearray]]],
    ) -> List[Part]:
        parts: List[Part] = []
        if not media_inputs:
            return parts

        for item in media_inputs:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    payload, mime_type = item
                    inferred_mime = str(mime_type) if mime_type else None
                    if isinstance(payload, str):
                        parts.extend(
                            self._prepare_media_parts(
                                [
                                    {
                                        "uri": payload,
                                        "mime_type": inferred_mime or self._infer_mime_type(payload),
                                    }
                                ]
                            )
                        )
                    elif isinstance(payload, (bytes, bytearray)):
                        parts.append(
                            Part.from_data(
                                mime_type=inferred_mime or "application/octet-stream",
                                data=bytes(payload),
                            )
                        )
                    else:
                        continue
                    continue

                if isinstance(item, (bytes, bytearray)):
                    parts.append(
                        Part.from_data(
                            mime_type="application/octet-stream",
                            data=bytes(item),
                        )
                    )
                    continue

                if isinstance(item, str):
                    if item.startswith("gs://") or item.startswith("http"):
                        mime_type = self._infer_mime_type(item) or "application/pdf"
                        parts.append(Part.from_uri(item, mime_type=mime_type))
                        continue

                    if os.path.exists(item):
                        mime_type = self._infer_mime_type(item) or "application/octet-stream"
                        parts.append(
                            Part.from_data(
                                mime_type=mime_type,
                                data=Path(item).read_bytes(),
                            )
                        )
                        continue

                    # Attempt to treat as base64 if not a path/URI
                    try:
                        decoded = base64.b64decode(item, validate=True)
                    except (binascii.Error, ValueError):
                        logger.warning("Unsupported media string input; skipping")
                        continue

                    parts.append(
                        Part.from_data(
                            mime_type="application/octet-stream",
                            data=decoded,
                        )
                    )
                    continue

                if isinstance(item, dict):
                    uri = item.get("uri")
                    mime_type = item.get("mime_type")
                    if uri:
                        parts.append(
                            Part.from_uri(
                                uri,
                                mime_type=mime_type or self._infer_mime_type(uri) or "application/pdf",
                            )
                        )
                        continue

                    base64_payload = item.get("base64") or item.get("data")
                    if base64_payload is None:
                        logger.warning("Media dict missing 'uri' or 'data'; skipping")
                        continue

                    if isinstance(base64_payload, (bytes, bytearray)):
                        decoded_bytes = bytes(base64_payload)
                    elif isinstance(base64_payload, str):
                        try:
                            decoded_bytes = base64.b64decode(base64_payload, validate=True)
                        except (binascii.Error, ValueError):
                            decoded_bytes = base64_payload.encode("utf-8")
                    else:
                        logger.warning("Unsupported media payload type %s", type(base64_payload))
                        continue

                    parts.append(
                        Part.from_data(
                            mime_type=mime_type or "application/octet-stream",
                            data=decoded_bytes,
                        )
                    )
                    continue

                logger.warning("Unsupported media input type %s; skipping", type(item).__name__)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to prepare media input (%s): %s", type(item).__name__, exc)

        return parts

    async def _legacy_summarize_pitch_deck(
        self,
        full_text: str,
        media_inputs: Optional[Sequence[Union[str, Tuple[Any, str], Dict[str, Any], bytes, bytearray]]] = None,
    ) -> Dict[str, Any]:
        summary_prompt = (
            "Analyze the following pitch deck content and extract information for these sections:\n"
            "- problem: What problem is being solved?\n"
            "- solution: What is the proposed solution?\n"
            "- market: Market size, opportunity, and target customers\n"
            "- team: Information about the founding team and key personnel\n"
            "- traction: Current progress, metrics, customers, revenue\n"
            "- financials: Financial projections, funding requirements, revenue model\n\n"
            "You must transcribe quantitative data from charts or tables when referenced and note any design cues that reveal the startup's maturity.\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return the analysis as a JSON object with the above keys. Be concise but comprehensive.\n"
            'If a section is not clearly addressed in the pitch deck, indicate "Not specified" for that key.'
        )

        summary_text = self.generate_text(summary_prompt, media_inputs)

        founder_prompt = (
            "Analyze the following pitch deck content and extract list of founders in array:\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return the analysis as an array.\n"
            "If no data found send empty array."
        )

        founder_raw = self.generate_text(founder_prompt, media_inputs)
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

        sector_response = self._strip_json_fences(self.generate_text(sector_prompt, media_inputs))

        company_name_prompt = (
            "Analyze the following pitch deck content and extract name of the startup/company this pitch is for:\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            "Return the organization or company name only with no extra words.\n"
            'If no data found send empty string "".'
        )

        company_name_response = self._strip_json_fences(self.generate_text(company_name_prompt, media_inputs))

        product_name_prompt = (
            "Analyze the following pitch deck content and extract the primary product or platform name the startup is promoting.\n\n"
            "Pitch deck content:\n"
            f"{full_text}\n\n"
            'Return the product/solution name only with no extra words. If none is mentioned, return an empty string "".'
        )

        product_name_response = self._strip_json_fences(self.generate_text(product_name_prompt, media_inputs))

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
        media_inputs: Optional[Sequence[Union[str, Tuple[Any, str], Dict[str, Any], bytes, bytearray]]] = None,
    ) -> Dict[str, Any]:
        """Summarize pitch deck into structured sections."""
        try:
            prompt = (
                "You are processing a startup pitch deck. Read the content below and return a JSON object with the following keys:\n"
                '- "summary": concise overall summary (string)\n'
                '- "founders": array of founder names (strings only)\n'
                '- "sector": specific industry or sector (string)\n'
                '- "company_name": organization or legal company name (string)\n'
                '- "product_name": primary product/solution or brand being pitched (string, may be empty)\n\n'
                "Pitch deck content:\n"
                f"{full_text}\n\n"
                "Requirements:\n"
                "- The JSON must be valid and parsable with standard libraries.\n"
                '- If only a product or brand name is mentioned, put it in both "company_name" and "product_name".\n'
                '- If both company and product are mentioned, ensure the company/legal entity goes in "company_name" and the flagship product/solution goes in "product_name".\n'
                '- Explicitly transcribe quantitative metrics that appear in charts or tables (ARR, MRR, revenue, funding, runway, valuations) into plain text values.\n'
                '- Mention key observations about slide design or visual polish in the "summary" output when they reveal business maturity.\n'
                '- Trim whitespace from values and avoid commentary.\n'
                '- When unsure, leave the value as an empty string "".'
            )

            structured_raw = self.generate_text(prompt, media_inputs)
            structured_clean = self._strip_json_fences(structured_raw)

            try:
                structured_payload: Dict[str, Any] = json.loads(structured_clean) if structured_clean else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse structured summary payload; falling back to legacy prompts")
                return await self._legacy_summarize_pitch_deck(full_text, media_inputs)

            summary_text = str(structured_payload.get("summary", "")).strip()
            founder_response = self._dedupe_preserve_order(
                self._coerce_string_list(structured_payload.get("founders", []))
            )
            sector_response = str(structured_payload.get("sector", "")).strip()
            company_name_response = str(structured_payload.get("company_name", "")).strip()
            product_name_response = str(structured_payload.get("product_name", "")).strip()

            if not summary_text:
                summary_text = self.generate_text(
                    "Provide a concise summary of the following pitch deck in under 160 words:\n"
                    f"{full_text}",
                    media_inputs,
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

            return self.generate_text(prompt)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Audio summarization error: %s", exc)
            return "Error processing audio transcript"

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
                9. When pitch materials include charts or tables, extract the quantitative values explicitly so that funding, revenue, ARR/MRR, runway, and valuation fields are populated instead of "Not available".
                10. Use observations about slide design, visual polish, and branding consistency to infer business maturity and reflect that insight in the appropriate commentary fields (e.g., company overview, financial rationale, or conclusion).

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

            raw_files = deal_data.get("raw_files", {})
            media_inputs: List[Union[str, Tuple[Any, str], Dict[str, Any], bytes, bytearray]] = []

            pitch_deck_uri = raw_files.get("pitch_deck_url") if isinstance(raw_files, dict) else None
            if isinstance(pitch_deck_uri, str) and pitch_deck_uri:
                media_inputs.append({"uri": pitch_deck_uri, "mime_type": "application/pdf"})

            video_uri = raw_files.get("video_pitch_deck_url") if isinstance(raw_files, dict) else None
            if isinstance(video_uri, str) and video_uri:
                media_inputs.append({"uri": video_uri, "mime_type": "video/mp4"})

            audio_uri = raw_files.get("audio_pitch_deck_url") if isinstance(raw_files, dict) else None
            if isinstance(audio_uri, str) and audio_uri:
                media_inputs.append({"uri": audio_uri, "mime_type": "audio/mpeg"})

            text_uri = raw_files.get("text_pitch_deck_url") if isinstance(raw_files, dict) else None
            if isinstance(text_uri, str) and text_uri:
                media_inputs.append({"uri": text_uri, "mime_type": "text/plain"})

            raw_response = self._generate_text(
                prompt,
                media_parts=self._prepare_media_parts(media_inputs),
            )
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

            raw_pages = pitch_deck.get("raw")
            if isinstance(raw_pages, dict) and raw_pages:
                numeric_snippets: List[str] = []
                seen_snippets: set[str] = set()
                sorted_pages = sorted(raw_pages.items(), key=lambda item: item[0])
                for page_label, page_text in sorted_pages:
                    if not isinstance(page_text, str):
                        continue
                    for line in page_text.splitlines():
                        normalized = line.strip()
                        if not normalized:
                            continue
                        if not re.search(r"\d", normalized):
                            continue
                        key = normalized.lower()
                        if key in seen_snippets:
                            continue
                        seen_snippets.add(key)
                        snippet = f"Page {page_label}: {normalized}"
                        numeric_snippets.append(snippet[:300])
                        if len(numeric_snippets) >= 40:
                            break
                    if len(numeric_snippets) >= 40:
                        break

                if numeric_snippets:
                    context_parts.append("Key numeric snippets from pitch deck (page: detail):")
                    context_parts.extend(numeric_snippets)

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
