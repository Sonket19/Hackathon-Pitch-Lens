"""Utilities for grounded retrieval using Vertex AI."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ToolConfig,
    grounding,
)

from config.settings import settings


logger = logging.getLogger(__name__)


class GroundedKnowledgeAgent:
    """Simple helper around Gemini with Google Search / Vertex Search grounding."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        enable_google_search: Optional[bool] = None,
        datastore: Optional[str] = None,
    ) -> None:
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

        chosen_model = model_name or settings.VERTEX_GROUNDED_MODEL
        self._model = GenerativeModel(chosen_model)
        self._config = GenerationConfig(
            temperature=0.15,
            top_p=0.8,
            top_k=32,
            max_output_tokens=1024,
        )

        tools: List[Tool] = []
        datastore_id = datastore if datastore is not None else settings.VERTEX_GROUNDED_DATASTORE
        if datastore_id:
            try:
                vertex_search = grounding.Retrieval(
                    grounding.VertexAISearch(datastore=datastore_id)
                )
                tools.append(Tool.from_retrieval(vertex_search))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Unable to initialise Vertex AI Search grounding: %s", exc)

        use_google = settings.VERTEX_ENABLE_GOOGLE_GROUNDING if enable_google_search is None else enable_google_search
        if use_google:
            try:
                google_search = grounding.GoogleSearchRetrieval()
                tools.append(Tool.from_google_search_retrieval(google_search))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Unable to initialise Google Search grounding: %s", exc)

        if tools:
            self._tools = tools
            self._tool_config: Optional[ToolConfig] = ToolConfig(
                function_calling_config=ToolConfig.FunctionCallingConfig(
                    mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                )
            )
        else:
            self._tools = None
            self._tool_config = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def tools(self) -> Optional[List[Tool]]:
        return self._tools

    @property
    def tool_config(self) -> Optional[ToolConfig]:
        return self._tool_config

    def grounded_completion(
        self,
        prompt: str,
        *,
        context_parts: Optional[Sequence[str]] = None,
    ) -> str:
        """Execute a grounded prompt, returning raw text."""

        contents: List[Part | str] = [prompt]
        for item in context_parts or ():
            if isinstance(item, Part):
                contents.append(item)
            elif isinstance(item, str) and item.strip():
                contents.append(item.strip())

        if len(contents) == 1:
            contents = [contents[0]]

        response = self._model.generate_content(
            contents if len(contents) > 1 else contents[0],
            generation_config=self._config,
            tools=self._tools,
            tool_config=self._tool_config,
        )

        return self._extract_text(response)

    def market_intel_snapshot(
        self,
        company_name: str,
        sector: str,
        *,
        founder_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Collect grounded market sizing, competitor, and news intel as JSON."""

        founders = ", ".join(str(name).strip() for name in (founder_names or []) if str(name).strip())
        instructions = f"""
            You are a capital markets research analyst. Produce a grounded fact-pack for due diligence.\n
            Company under review: {company_name or 'Unknown'}\n
            Sector focus: {sector or 'Unknown'}\n
            Founding team: {founders or 'Unknown'}\n
            Return a strict JSON object with the following shape:\n
            {{
              "market_stats": {{
                "tam": {{"label": "string", "value": "string", "cagr": "string", "source": "string"}},
                "som": {{"label": "string", "value": "string", "cagr": "string", "source": "string"}},
                "commentary": "string"
              }},
              "competitors": [{{"name": "string", "description": "string", "source": "string"}}],
              "recent_news": [{{"headline": "string", "date": "string", "source": "string", "url": "string"}}]
            }}\n
            Ensure every metric is backed by grounded evidence. When the corpus provides citations, include the originating URL
            in the "source" or "url" fields. Use "Not available" when data cannot be verified.
        """

        raw = self.grounded_completion(instructions)
        payload = self._load_json(raw)
        if isinstance(payload, dict):
            return payload
        return {
            "market_stats": {
                "tam": {"label": "Not available", "value": "Not available", "cagr": "Not available", "source": "Not available"},
                "som": {"label": "Not available", "value": "Not available", "cagr": "Not available", "source": "Not available"},
                "commentary": "Not available",
            },
            "competitors": [],
            "recent_news": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_text(self, response: Any) -> str:  # type: ignore[override]
        text = getattr(response, "text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

        chunks: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) if content else []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)
        return "".join(chunks)

    @staticmethod
    def _load_json(raw: str) -> Any:
        cleaned = raw.strip()
        if not cleaned:
            return {}
        cleaned = GroundedKnowledgeAgent._strip_json_fence(cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("Failed to decode grounded response as JSON: %s", cleaned)
            return {}

    @staticmethod
    def _strip_json_fence(payload: str) -> str:
        return payload.strip().removeprefix("```json").removesuffix("```").strip()

