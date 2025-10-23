from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiSummarizer:
    """Wrapper around Gemini with deterministic defaults and parsing helpers."""

    def __init__(self) -> None:
        self.model: Optional[GenerativeModel] = None
        # Force deterministic behaviour so repeated uploads stay consistent.
        self._generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )

        try:
            if settings.GCP_PROJECT_ID:
                vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
                self.model = GenerativeModel("gemini-2.5-pro")
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.warning("Gemini unavailable (%s); using local summariser", exc)

    def _generate_text(self, prompt: str) -> str:
        if not self.model:
            return self._fallback_generate_text(prompt)

        response = self.model.generate_content(
            prompt,
            generation_config=self._generation_config,
        )
        text = getattr(response, "text", "")
        return text.strip() if isinstance(text, str) else ""

    def generate_text(self, prompt: str) -> str:
        """Public wrapper for deterministic text generation."""
        return self._generate_text(prompt)

    def _fallback_generate_text(self, prompt: str) -> str:
        """Generate deterministic text without Vertex AI.

        The prompts usually end with the corpus to summarise. We extract the
        final paragraph and surface a concise snippet so downstream callers still
        receive meaningful information even when Gemini is unavailable.
        """

        text = prompt.split("\n\n")[-1].strip() or prompt
        sentences = re.split(r"(?<=[.!?])\s+", text)
        summary = " ".join(sentences[:3]).strip()
        return summary or textwrap.shorten(text, width=280, placeholder="...")

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

    async def summarize_pitch_deck(self, full_text: str) -> Dict[str, Any]:
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
                '- Trim whitespace from values and avoid commentary.\n'
                '- When unsure, leave the value as an empty string "".'
            )

            if not self.model:
                return await self._local_summarize_pitch_deck(full_text)

            structured_raw = self._generate_text(prompt)
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
                    "Provide a concise summary of the following pitch deck in under 160 words:\n"
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
            return await self._local_summarize_pitch_deck(full_text)

    async def _local_summarize_pitch_deck(self, full_text: str) -> Dict[str, Any]:
        """Heuristic summariser used when Gemini is unavailable."""

        cleaned_lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        joined_text = " ".join(cleaned_lines)
        sentences = re.split(r"(?<=[.!?])\s+", joined_text)
        summary = " ".join(sentences[:5]).strip()
        if not summary:
            summary = textwrap.shorten(joined_text or "No content supplied.", width=320, placeholder="...")

        founder_candidates: List[str] = []
        for line in cleaned_lines:
            if re.search(r"(?i)founder|ceo|team", line):
                founder_candidates.extend(
                    re.findall(r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+", line)
                )
        founders = self._dedupe_preserve_order(founder_candidates)[:3]

        sector = ""
        lowered = joined_text.lower()
        sector_keywords = {
            "artificial intelligence": ["artificial intelligence", "ai", "machine learning"],
            "fintech": ["fintech", "payments", "banking"],
            "healthcare": ["health", "healthcare", "medtech", "biotech"],
            "climate": ["climate", "sustainability", "energy"],
        }
        for label, keywords in sector_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                sector = label
                break
        if not sector and cleaned_lines:
            sector = cleaned_lines[0].split(" ")[0]

        company_name = ""
        if cleaned_lines:
            headline = cleaned_lines[0]
            match = re.findall(r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+", headline)
            if match:
                company_name = match[0]

        product_name = ""
        if len(cleaned_lines) > 1:
            second_line = cleaned_lines[1]
            product_match = re.findall(r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+", second_line)
            if product_match:
                product_name = product_match[0]

        return {
            "summary_res": summary,
            "founder_response": founders,
            "sector_response": sector,
            "company_name_response": company_name,
            "product_name_response": product_name,
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

    async def generate_memo(self, deal_data: Dict[str, Any], weightage: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete investment memo."""
        try:
            metadata = deal_data.get("metadata", {})
            extracted_text = deal_data.get("extracted_text", {})
            public_data = deal_data.get("public_data", {})
            user_input = deal_data.get("user_input", {})

            context = self._build_memo_context(metadata, extracted_text, public_data, user_input)

            if not self.model:
                return self._build_stub_memo(metadata, extracted_text, public_data)

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

    def _build_stub_memo(
        self,
        metadata: Dict[str, Any],
        extracted_text: Dict[str, Any],
        public_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a lightweight memo when Gemini is unavailable."""

        pitch_summary = ""
        if isinstance(extracted_text, dict):
            pitch = extracted_text.get("pitch_deck")
            if isinstance(pitch, dict):
                pitch_summary = str(pitch.get("concise", ""))

        def _fallback(value: Any, default: str = "Not available") -> str:
            if isinstance(value, str) and value.strip():
                return value.strip()
            return default

        founders = []
        if isinstance(metadata.get("founder_names"), list):
            founders = [str(item) for item in metadata["founder_names"] if str(item).strip()]

        return {
            "company_overview": {
                "name": _fallback(metadata.get("display_name") or metadata.get("company_name")),
                "sector": _fallback(metadata.get("sector")),
                "founders": founders,
                "technology": pitch_summary or "Not available",
            },
            "market_analysis": {
                "industry_size_and_growth": {
                    "commentary": _fallback(public_data.get("market_stats", {}).get("summary") if isinstance(public_data.get("market_stats"), dict) else ""),
                },
                "recent_news": _fallback("; ".join(public_data.get("news", [])) if isinstance(public_data.get("news"), list) else ""),
            },
            "business_model": {
                "revenue_streams": "Not available",
                "pricing": "Not available",
                "scalability": "Not available",
                "unit_economics": {},
            },
            "financials": {},
            "claims_analysis": [],
            "risk_metrics": {},
            "conclusion": {
                "overall_attractiveness": "Not available",
            },
        }

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
