from __future__ import annotations

import json
import logging
import re
import base64
import binascii
import mimetypes
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, Part

from config.settings import settings

logger = logging.getLogger(__name__)


MAX_CONTEXT_LENGTH = 15000


DEFAULT_MEMO_TEMPLATE: Dict[str, Any] = {
    "company_overview": {
        "name": "Not available",
        "sector": "Not available",
        "founders": [],
        "technology": "Not available",
    },
    "market_analysis": {
        "industry_size_and_growth": {
            "total_addressable_market": {
                "name": "Not available",
                "value": "Not available",
                "cagr": "Not available",
                "source": "Not available",
            },
            "serviceable_obtainable_market": {
                "name": "Not available",
                "value": "Not available",
                "cagr": "Not available",
                "source": "Not available",
            },
            "commentary": "Not available",
        },
        "recent_news": "Not available",
        "competitor_details": [],
        "sub_segment_opportunities": [],
    },
    "business_model": {
        "revenue_streams": "Not available",
        "pricing": "Not available",
        "scalability": "Not available",
        "unit_economics": {
            "customer_lifetime_value_ltv": "Not available",
            "customer_acquisition_cost_cac": "Not available",
        },
    },
    "financials": {
        "funding_history": "Not available",
        "projections": [],
        "valuation_rationale": "Not available",
        "srr_mrr": {
            "current_booked_arr": "Not available",
            "current_mrr": "Not available",
        },
        "burn_and_runway": {
            "funding_ask": "Not available",
            "stated_runway": "Not available",
            "implied_net_burn": "Not available",
        },
    },
    "claims_analysis": [],
    "risk_metrics": {
        "composite_risk_score": 0,
        "score_interpretation": "Not available",
        "narrative_justification": "Not available",
    },
    "conclusion": {
        "overall_attractiveness": "Not available",
    },
}


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
        """Send a prompt to Gemini, retrying without media if multimodal fails."""

        def _build_content(parts: Optional[List[Part]]) -> Union[str, List[Union[str, Part]]]:
            if parts:
                try:
                    prompt_part: Union[str, Part] = Part.from_text(prompt)  # type: ignore[attr-defined]
                    return [prompt_part, *parts]
                except AttributeError:
                    return [prompt, *parts]
            return prompt

        try:
            response = self.model.generate_content(
                _build_content(media_parts),
                generation_config=self._generation_config,
            )
        except Exception as exc:
            if media_parts:
                logger.warning(
                    "Multimodal Gemini call failed (%s); retrying with text-only prompt.",
                    exc,
                )
                response = self.model.generate_content(
                    prompt,
                    generation_config=self._generation_config,
                )
            else:
                raise

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

            if len(context) > MAX_CONTEXT_LENGTH:
                context = context[:MAX_CONTEXT_LENGTH]

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

            media_parts = self._prepare_media_parts(media_inputs)
            raw_response = self._generate_text(
                prompt,
                media_parts=media_parts,
            )
            clean = re.sub(r"^```json\s*|\s*```$", "", raw_response, flags=re.MULTILINE).strip()

            if clean:
                try:
                    parsed = json.loads(clean)
                    if isinstance(parsed, dict):
                        parsed = self._fill_financial_placeholders(parsed, context, media_parts)
                        merged = self._merge_with_template(parsed)
                        return self._apply_context_overrides(
                            merged,
                            metadata,
                            extracted_text,
                            public_data,
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        "Gemini memo response was not valid JSON; falling back to default template.",
                    )

            fallback_seed = self._fill_financial_placeholders({}, context, media_parts)
            fallback = self._merge_with_template(fallback_seed)
            contextualised = self._apply_context_overrides(
                fallback,
                metadata,
                extracted_text,
                public_data,
            )
            if clean:
                contextualised["raw_text"] = clean
            return contextualised

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

    @staticmethod
    def _merge_with_template(payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(DEFAULT_MEMO_TEMPLATE)

        def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    _deep_update(target[key], value)
                else:
                    target[key] = value

        _deep_update(merged, payload)
        return merged

    def _collect_missing_financial_fields(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        missing_structure: Dict[str, Any] = {}
        missing_paths: List[str] = []

        financials = payload.get("financials")
        if not isinstance(financials, dict):
            missing_structure = {
                "funding_history": "",
                "valuation_rationale": "",
                "projections": [{"year": "", "revenue": ""}],
                "srr_mrr": {"current_booked_arr": "", "current_mrr": ""},
                "burn_and_runway": {
                    "funding_ask": "",
                    "stated_runway": "",
                    "implied_net_burn": "",
                },
            }
            missing_paths.extend(
                [
                    "financials.funding_history",
                    "financials.valuation_rationale",
                    "financials.projections",
                    "financials.srr_mrr.current_booked_arr",
                    "financials.srr_mrr.current_mrr",
                    "financials.burn_and_runway.funding_ask",
                    "financials.burn_and_runway.stated_runway",
                    "financials.burn_and_runway.implied_net_burn",
                ]
            )
            return missing_structure, missing_paths

        def _mark_missing(
            container: Dict[str, Any],
            key: str,
            placeholder: Any,
            path: str,
        ) -> None:
            value = container.get(key)
            if key not in missing_structure and self._is_placeholder(value):
                missing_structure[key] = placeholder
                missing_paths.append(path)

        _mark_missing(financials, "funding_history", "", "financials.funding_history")
        _mark_missing(financials, "valuation_rationale", "", "financials.valuation_rationale")

        projections_value = financials.get("projections")
        if self._is_placeholder(projections_value):
            missing_structure["projections"] = [{"year": "", "revenue": ""}]
            missing_paths.append("financials.projections")

        srr_mrr_value = financials.get("srr_mrr")
        srr_missing: Dict[str, Any] = {}
        srr_paths: List[str] = []
        if not isinstance(srr_mrr_value, dict):
            srr_missing = {"current_booked_arr": "", "current_mrr": ""}
            srr_paths.extend(
                [
                    "financials.srr_mrr.current_booked_arr",
                    "financials.srr_mrr.current_mrr",
                ]
            )
        else:
            if self._is_placeholder(srr_mrr_value.get("current_booked_arr")):
                srr_missing["current_booked_arr"] = ""
                srr_paths.append("financials.srr_mrr.current_booked_arr")
            if self._is_placeholder(srr_mrr_value.get("current_mrr")):
                srr_missing["current_mrr"] = ""
                srr_paths.append("financials.srr_mrr.current_mrr")

        if srr_missing:
            missing_structure["srr_mrr"] = srr_missing
            missing_paths.extend(srr_paths)

        burn_value = financials.get("burn_and_runway")
        burn_missing: Dict[str, Any] = {}
        if not isinstance(burn_value, dict):
            burn_missing = {
                "funding_ask": "",
                "stated_runway": "",
                "implied_net_burn": "",
            }
        else:
            for key in ("funding_ask", "stated_runway", "implied_net_burn"):
                if self._is_placeholder(burn_value.get(key)):
                    burn_missing[key] = ""
                    missing_paths.append(f"financials.burn_and_runway.{key}")

        if burn_missing:
            missing_structure["burn_and_runway"] = burn_missing

        # Deduplicate missing paths while preserving order
        seen_paths: set[str] = set()
        unique_paths: List[str] = []
        for item in missing_paths:
            if item in seen_paths:
                continue
            seen_paths.add(item)
            unique_paths.append(item)

        return missing_structure, unique_paths

    def _fill_financial_placeholders(
        self,
        payload: Dict[str, Any],
        context: str,
        media_parts: Optional[List[Part]],
    ) -> Dict[str, Any]:
        missing_structure, missing_paths = self._collect_missing_financial_fields(payload)
        if not missing_paths:
            return payload

        prompt_structure = json.dumps({"financials": missing_structure}, indent=2)
        fields_list = ", ".join(missing_paths)

        followup_prompt = (
            "You previously drafted an investment memo but some financial fields were left blank.\n"
            "Fill ONLY the missing fields listed below using the startup materials provided.\n"
            "If a field is truly absent from the materials, respond with 'Not available'.\n"
            "Return strict JSON matching the structure.\n\n"
            f"Missing fields: {fields_list}\n\n"
            "Startup materials:\n"
            f"{context}\n\n"
            "JSON structure to populate:\n"
            f"{prompt_structure}\n"
        )

        followup_raw = self._generate_text(followup_prompt, media_parts=media_parts)
        followup_clean = self._strip_json_fences(followup_raw)

        if not followup_clean:
            return payload

        try:
            parsed = json.loads(followup_clean)
        except json.JSONDecodeError:
            logger.warning("Financial follow-up response was not valid JSON")
            return payload

        if not isinstance(parsed, dict):
            return payload

        updates = parsed.get("financials") if isinstance(parsed.get("financials"), dict) else parsed
        if not isinstance(updates, dict):
            return payload

        financial_section = payload.setdefault("financials", {})

        def _merge_missing(target: Dict[str, Any], update_payload: Dict[str, Any]) -> None:
            for key, value in update_payload.items():
                if isinstance(value, dict):
                    nested_target = target.get(key)
                    if not isinstance(nested_target, dict):
                        nested_target = {}
                    _merge_missing(nested_target, value)
                    if nested_target and not self._is_placeholder(nested_target):
                        target[key] = nested_target
                    elif key not in target:
                        target[key] = nested_target
                elif isinstance(value, list):
                    if value and not self._is_placeholder(value):
                        target[key] = value
                else:
                    if not self._is_placeholder(value):
                        existing_value = target.get(key)
                        if self._is_placeholder(existing_value):
                            target[key] = value
                        elif key not in target:
                            target[key] = value

        _merge_missing(financial_section, updates)
        return payload

    @staticmethod
    def _is_placeholder(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            normalised = value.strip().lower()
            return normalised in {"", "not available", "n/a", "not specified", "unknown"}
        if isinstance(value, (list, dict)):
            return not value
        return False

    def _apply_context_overrides(
        self,
        memo: Dict[str, Any],
        metadata: Dict[str, Any],
        extracted_text: Dict[str, Any],
        public_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        memo = deepcopy(memo)
        metadata = metadata if isinstance(metadata, dict) else {}
        extracted_text = extracted_text if isinstance(extracted_text, dict) else {}
        public_data = public_data if isinstance(public_data, dict) else {}

        def _assign_if_placeholder(container: Dict[str, Any], key: str, new_value: Any) -> None:
            if new_value is None:
                return
            if key not in container or self._is_placeholder(container.get(key)):
                container[key] = new_value

        company_overview = memo.get("company_overview", {})
        if not isinstance(company_overview, dict):
            company_overview = {}
            memo["company_overview"] = company_overview

        company_name = metadata.get("display_name") or metadata.get("company_name") or metadata.get("product_name")
        if isinstance(company_name, str) and company_name.strip():
            _assign_if_placeholder(company_overview, "name", company_name.strip())

        sector = metadata.get("sector")
        if isinstance(sector, str) and sector.strip():
            _assign_if_placeholder(company_overview, "sector", sector.strip())

        founders_section = company_overview.get("founders")
        if not isinstance(founders_section, list):
            founders_section = [] if founders_section is None else [founders_section]
        founder_names = metadata.get("founder_names")
        if isinstance(founder_names, list) and founder_names:
            existing_lookup: set[str] = set()
            for item in founders_section:
                if isinstance(item, dict):
                    name_value = str(item.get("name", "")).strip()
                else:
                    name_value = str(item).strip()
                if name_value:
                    existing_lookup.add(name_value.lower())
            enhanced_founders: List[Dict[str, str]] = []
            for entry in founders_section:
                if isinstance(entry, dict):
                    name_value = str(entry.get("name", "")).strip()
                else:
                    name_value = str(entry).strip()
                if not name_value:
                    continue
                professional_background = "Not available"
                education = "Not available"
                previous_ventures = "Not available"
                if isinstance(entry, dict):
                    professional_background = entry.get("professional_background", professional_background)
                    education = entry.get("education", education)
                    previous_ventures = entry.get("previous_ventures", previous_ventures)
                enhanced_founders.append(
                    {
                        "name": name_value,
                        "education": education,
                        "professional_background": professional_background,
                        "previous_ventures": previous_ventures,
                    }
                )
            for name in founder_names:
                if not isinstance(name, str) or not name.strip():
                    continue
                lowered = name.strip().lower()
                if lowered in existing_lookup:
                    continue
                enhanced_founders.append(
                    {
                        "name": name.strip(),
                        "education": "Not available",
                        "professional_background": "Not available",
                        "previous_ventures": "Not available",
                    }
                )
                existing_lookup.add(lowered)
            if enhanced_founders:
                company_overview["founders"] = enhanced_founders

        founder_profile = public_data.get("founder_profile")
        if isinstance(founder_profile, str) and founder_profile.strip() and company_overview.get("founders"):
            for founder in company_overview["founders"]:
                if isinstance(founder, dict) and self._is_placeholder(founder.get("professional_background")):
                    founder["professional_background"] = founder_profile.strip()

        market_analysis = memo.get("market_analysis", {})
        if not isinstance(market_analysis, dict):
            market_analysis = {}
            memo["market_analysis"] = market_analysis

        market_stats = public_data.get("market_stats")
        if isinstance(market_stats, dict) and market_stats:
            industry = market_analysis.get("industry_size_and_growth")
            if not isinstance(industry, dict):
                industry = {}
                market_analysis["industry_size_and_growth"] = industry

            tam_section = industry.get("total_addressable_market")
            if not isinstance(tam_section, dict):
                tam_section = {"name": "Not available", "value": "Not available", "cagr": "Not available", "source": "Not available"}
                industry["total_addressable_market"] = tam_section

            sam_section = industry.get("serviceable_obtainable_market")
            if not isinstance(sam_section, dict):
                sam_section = {"name": "Not available", "value": "Not available", "cagr": "Not available", "source": "Not available"}
                industry["serviceable_obtainable_market"] = sam_section

            tam_value = market_stats.get("TAM") or market_stats.get("tam")
            sam_value = market_stats.get("SAM") or market_stats.get("sam")
            cagr_value = market_stats.get("CAGR") or market_stats.get("cagr")
            trend_value = market_stats.get("key_trends") or market_stats.get("trends")

            if isinstance(tam_value, dict):
                for field in ("name", "value", "cagr", "source"):
                    if field in tam_value:
                        _assign_if_placeholder(tam_section, field, tam_value[field])
            elif isinstance(tam_value, str) and tam_value.strip():
                _assign_if_placeholder(tam_section, "value", tam_value.strip())
                if self._is_placeholder(tam_section.get("name")):
                    tam_section["name"] = f"{sector or 'Market'} TAM".strip()

            if isinstance(sam_value, dict):
                for field in ("name", "value", "cagr", "source", "projection"):
                    if field in sam_value:
                        _assign_if_placeholder(sam_section, field, sam_value[field])
            elif isinstance(sam_value, str) and sam_value.strip():
                _assign_if_placeholder(sam_section, "value", sam_value.strip())
                if self._is_placeholder(sam_section.get("name")):
                    sam_section["name"] = f"{sector or 'Market'} SOM".strip()

            if isinstance(cagr_value, str) and cagr_value.strip():
                _assign_if_placeholder(tam_section, "cagr", cagr_value.strip())
                _assign_if_placeholder(sam_section, "cagr", cagr_value.strip())

            if isinstance(trend_value, (list, tuple)):
                opportunities = [str(item).strip() for item in trend_value if str(item).strip()]
            elif isinstance(trend_value, str):
                opportunities = [part.strip() for part in re.split(r"[•\-\n]+", trend_value) if part.strip()]
            else:
                opportunities = []

            if opportunities and self._is_placeholder(market_analysis.get("sub_segment_opportunities")):
                market_analysis["sub_segment_opportunities"] = opportunities

        competitors = public_data.get("competitors")
        if isinstance(competitors, list) and competitors:
            competitor_details = []
            for entry in competitors:
                if isinstance(entry, dict):
                    name = str(entry.get("name") or entry.get("company") or entry.get("title") or "").strip()
                    commentary = str(entry.get("commentary") or entry.get("description") or "Not available").strip()
                    business_model = str(entry.get("business_model") or entry.get("model") or "Not available").strip()
                    funding = str(entry.get("funding") or entry.get("funding_stage") or "Not available").strip()
                    margins = str(entry.get("margins") or entry.get("growth") or "Not available").strip()
                else:
                    name = str(entry).strip()
                    commentary = "Identified via public web search"
                    business_model = "Not available"
                    funding = "Not available"
                    margins = "Not available"
                if not name:
                    continue
                competitor_details.append(
                    {
                        "name": name,
                        "commentary": commentary or "Not available",
                        "business_model": business_model or "Not available",
                        "funding": funding or "Not available",
                        "margins": margins or "Not available",
                    }
                )
            if competitor_details and self._is_placeholder(market_analysis.get("competitor_details")):
                market_analysis["competitor_details"] = competitor_details

        news_items = public_data.get("news")
        if isinstance(news_items, list) and news_items:
            joined_news = "\n".join(str(item).strip() for item in news_items if str(item).strip())
            if joined_news:
                _assign_if_placeholder(market_analysis, "recent_news", joined_news)

        financials = memo.get("financials", {})
        if not isinstance(financials, dict):
            financials = {}
            memo["financials"] = financials

        extracted_metrics = self._extract_financial_metrics(extracted_text)

        srr_mrr = financials.get("srr_mrr")
        if not isinstance(srr_mrr, dict):
            srr_mrr = {"current_booked_arr": "Not available", "current_mrr": "Not available"}
            financials["srr_mrr"] = srr_mrr

        for key in ("current_booked_arr", "current_mrr"):
            metric_value = extracted_metrics.get(key)
            if isinstance(metric_value, str) and metric_value.strip():
                _assign_if_placeholder(srr_mrr, key, metric_value.strip())

        burn_and_runway = financials.get("burn_and_runway")
        if not isinstance(burn_and_runway, dict):
            burn_and_runway = {
                "funding_ask": "Not available",
                "stated_runway": "Not available",
                "implied_net_burn": "Not available",
            }
            financials["burn_and_runway"] = burn_and_runway

        for key in ("funding_ask", "stated_runway", "implied_net_burn"):
            metric_value = extracted_metrics.get(key)
            if isinstance(metric_value, str) and metric_value.strip():
                _assign_if_placeholder(burn_and_runway, key, metric_value.strip())

        for field in ("funding_history", "valuation_rationale"):
            metric_value = extracted_metrics.get(field)
            if isinstance(metric_value, str) and metric_value.strip():
                _assign_if_placeholder(financials, field, metric_value.strip())

        projections = extracted_metrics.get("projections")
        if (
            isinstance(projections, list)
            and projections
            and self._is_placeholder(financials.get("projections"))
        ):
            financials["projections"] = projections

        if self._is_placeholder(financials.get("funding_history")):
            public_funding = public_data.get("news") or []
            if isinstance(public_funding, list):
                funding_lines = [line for line in public_funding if any(keyword in line.lower() for keyword in ("funding", "raise", "investment"))]
                if funding_lines:
                    financials["funding_history"] = funding_lines[0]

        business_model = memo.get("business_model", {})
        if not isinstance(business_model, dict):
            business_model = {}
            memo["business_model"] = business_model

        concise_pitch = extracted_text.get("pitch_deck", {}).get("concise") if isinstance(extracted_text, dict) else None
        if isinstance(concise_pitch, str) and concise_pitch.strip():
            for key in ("revenue_streams", "pricing", "scalability"):
                if self._is_placeholder(business_model.get(key)):
                    business_model[key] = concise_pitch.strip()

        return memo

    def _extract_financial_metrics(self, extracted_text: Dict[str, Any]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if not isinstance(extracted_text, dict):
            return metrics

        pitch_deck = extracted_text.get("pitch_deck")
        if not isinstance(pitch_deck, dict):
            return metrics

        raw_pages = pitch_deck.get("raw")
        if not isinstance(raw_pages, dict) or not raw_pages:
            return metrics

        page_lines: List[str] = []
        for page_text in raw_pages.values():
            if not isinstance(page_text, str):
                continue
            for line in page_text.splitlines():
                clean_line = line.strip()
                if clean_line:
                    page_lines.append(clean_line)

        joined_text = "\n".join(page_lines)

        def _search_patterns(patterns: Sequence[str]) -> Optional[str]:
            for pattern in patterns:
                match = re.search(pattern, joined_text, flags=re.IGNORECASE)
                if match:
                    groups = [grp for grp in match.groups() if grp]
                    if groups:
                        return groups[0].strip()
                    return match.group(0).strip()
            return None

        currency_pattern = (
            r"([\$₹€£]?\s?(?:~|≈)?\s?\d[\d,\.]*\s?"
            r"(?:k|m|b|mn|bn|million|billion|crore|crores|cr|crs|lakh|lakhs|lc)?\s?"
            r"(?:usd|inr|sgd|eur|cad|aud|gbp|rs)?"
            r")"
        )

        metrics_map = {
            "current_booked_arr": [
                rf"\bbooked\s+arr[^\n:=]*[:=\-–]?\s*{currency_pattern}",
                rf"\barr\b[^\n]*[:=\-–]\s*{currency_pattern}",
                rf"annual recurring revenue[^\n]*{currency_pattern}",
            ],
            "current_mrr": [
                rf"\bmrr\b[^\n]*[:=\-–]\s*{currency_pattern}",
                rf"monthly recurring revenue[^\n]*{currency_pattern}",
            ],
            "funding_ask": [
                rf"(?:funding ask|seeking|raising|raise)[^\n]*[:=\-–]?\s*{currency_pattern}",
            ],
            "stated_runway": [
                r"runway[^\n]*(\d+\s*(?:months?|mos?|years?|yrs?))",
                r"runway[^\n]*(?:of|for)\s*(\d+\s*(?:months?|mos?|years?|yrs?))",
            ],
            "implied_net_burn": [
                rf"(?:burn rate|net burn)[^\n]*[:=\-–]?\s*{currency_pattern}",
            ],
            "funding_history": [
                r"(?:raised|secured|closed)[^\n]+(?:round|funding|investment)[^\n]*",
            ],
            "valuation_rationale": [
                r"valuation[^\n]+",
                rf"valued[^\n]*[:=\-–]?\s*{currency_pattern}",
            ],
        }

        for key, patterns in metrics_map.items():
            value = _search_patterns(patterns)
            if value:
                metrics[key] = value

        projections: List[Dict[str, str]] = []
        seen_years: set[str] = set()
        projection_pattern = re.compile(r"(20\d{2})[^\n]*" + currency_pattern, re.IGNORECASE)
        fy_projection_pattern = re.compile(r"(FY(?:20)?\d{2})[^\n]*" + currency_pattern, re.IGNORECASE)
        for line in page_lines:
            match = projection_pattern.search(line)
            if not match:
                match = fy_projection_pattern.search(line)
            if not match:
                continue
            year = match.group(1)
            revenue = match.group(2)
            if not year or not revenue:
                continue
            normalized_year = year.upper()
            if normalized_year in seen_years:
                continue
            seen_years.add(normalized_year)
            projections.append({"year": normalized_year, "revenue": revenue.strip()})

        if projections:
            metrics["projections"] = projections

        return metrics
