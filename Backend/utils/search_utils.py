from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
import logging
import re
import time

from utils.email_utils import extract_emails
from utils.summarizer import GeminiSummarizer
from utils.vector_db import query_index

logger = logging.getLogger(__name__)


def query_vector_search(query: str, source: str = "general", *, k: int = 5) -> List[Dict[str, Any]]:
    """Query the Vertex AI vector index and normalise the response."""

    if not query:
        return []

    try:
        results = query_index(query=query, source=source, k=k)
    except Exception as exc:  # pragma: no cover - network/service failures
        logger.error("Vector search error for %s: %s", query, exc)
        return []

    normalised: List[Dict[str, Any]] = []
    for item in results:
        metadata = item.get("metadata") if isinstance(item, dict) else {}
        if metadata is None:
            metadata = {}

        title = (
            metadata.get("title")
            or metadata.get("heading")
            or item.get("title")
            or metadata.get("source_title")
            or ""
        )
        snippet = (
            metadata.get("snippet")
            or metadata.get("summary")
            or item.get("snippet")
            or item.get("text")
            or ""
        )
        link = metadata.get("url") or metadata.get("link") or metadata.get("source_url") or item.get("link") or ""
        normalised.append(
            {
                "title": str(title),
                "snippet": str(snippet),
                "link": str(link),
                "score": item.get("score"),
                "metadata": metadata,
            }
        )

    return normalised


class PublicDataGatherer:
    def __init__(self, *, summarizer: Optional[GeminiSummarizer] = None, vector_query=None):
        self.summarizer = summarizer or GeminiSummarizer()
        self._vector_query = vector_query or query_vector_search

    async def gather_data(
        self,
        company_name: str,
        founder_name: List[str],
        sector: str,
        logos: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Gather public data about company, founder, and market"""
        try:
            start_time = time.perf_counter()
#             data = {}

#             # Founder profile
#             data['founder_profile'] = await self._search_founder_profile(founder_name)

#             # Competitors
#             data['competitors'] = await self._search_competitors(company_name, sector)

#             # Market stats
#             data['market_stats'] = await self._search_market_data(sector)

#             # Recent news
#             data['news'] = await self._search_news(company_name, founder_name)

#             return data
        
        
            logo_inputs = [
                str(item).strip()
                for item in (logos or [])
                if isinstance(item, str) and str(item).strip()
            ]

            tasks = [
                self._search_founder_profile(founder_name),
                self._search_competitors(company_name, sector),
                self._search_market_data(sector),
                self._search_news(company_name, founder_name),
            ]

            if logo_inputs:
                tasks.append(self._resolve_logo_companies(logo_inputs))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            founder_payload: Dict[str, Any]
            founder_summary: str
            founder_contacts: Dict[str, Any]
            founder_result = results[0]
            if isinstance(founder_result, Exception):
                founder_summary = "Error gathering founder info"
                founder_contacts = {}
            elif isinstance(founder_result, dict):
                founder_summary = str(founder_result.get('summary', '') or '').strip() or "No public information found"
                contacts_value = founder_result.get('contacts')
                founder_contacts = contacts_value if isinstance(contacts_value, dict) else {}
            else:
                founder_summary = str(founder_result) if founder_result is not None else "No public information found"
                founder_contacts = {}

            data: Dict[str, Any] = {
                'founder_profile': founder_summary,
                'competitors': results[1] if not isinstance(results[1], Exception) else [],
                'market_stats': results[2] if not isinstance(results[2], Exception) else {},
                'news': results[3] if not isinstance(results[3], Exception) else []
            }

            if founder_contacts:
                data['founder_contacts'] = founder_contacts

            if logo_inputs:
                logo_index = 4
                if len(results) > logo_index and not isinstance(results[logo_index], Exception):
                    data['logo_companies'] = results[logo_index]
                else:
                    data['logo_companies'] = []

            logger.info(
                "Public data gathering for %s completed in %.3fs",
                company_name or founder_name,
                time.perf_counter() - start_time
            )

            return data

        except Exception as e:
            logger.error(f"Public data gathering error: {str(e)}")
            return {}
        
    async def _search_founder_profile(self, founder_name: List[str]) -> Dict[str, Any]:
        """Search for founder background information and potential contact emails."""
        try:
            founder_combined = ", ".join(founder_name)
            queries = [
                f"{founder_combined} background experience",
                f"{founder_combined} LinkedIn profile career",
                f"{founder_combined} founder entrepreneur"
            ]

#             patterns = [
#                 "background experience",
#                 "LinkedIn profile career",
#                 "founder entrepreneur"
#             ]

#             queries = [f"{name} {pattern}" for name in founder_name for pattern in patterns]
            
            all_results: List[Dict[str, Any]] = []
            for query in queries:
                results = await self._perform_search(query, num_results=3, source="founder")
                all_results.extend(results)

            logger.debug("Founder search results: %s", all_results)
            # Summarize findings
            emails: List[str] = extract_emails(all_results)
            email_sources: List[str] = []
            for result in all_results:
                link = result.get('link') if isinstance(result, dict) else None
                if isinstance(link, str) and link.strip():
                    email_sources.append(link.strip())

            if all_results:
                combined_text = "".join([f"{r['title']}: {r['snippet']}" for r in all_results])
                summary_text = self.summarizer.generate_text(
                    f"Summarize the professional background of {founder_combined} based on {combined_text}"
                )
                logger.debug("Founder background summary: %s", summary_text)
            else:
                summary_text = "No public information found"

            return {
                'summary': summary_text,
                'contacts': {
                    'emails': emails,
                    'sources': email_sources,
                },
                'results': all_results,
            }

        except Exception as e:
            logger.error(f"Founder search error: {str(e)}")
            return {
                'summary': "Error gathering founder information",
                'contacts': {},
                'results': [],
            }

    async def _search_competitors(self, company_name: str, sector: str) -> List[str]:
        """Search for competitors in the same sector"""
        try:
            query = f"{sector} companies competitors startups"
            results = await self._perform_search(query, num_results=5, source="competitor")

            if not results:
                return []

            # Extract competitor names using Gemini
            logger.debug("Competitor search results: %s", results)
            combined_text = "".join([f"{r['title']}: {r['snippet']}" for r in results])
            response_text = self.summarizer.generate_text(
                f"Extract a list of company names that are competitors to {company_name} in the {sector} sector from: {combined_text}. Return only company names, one per line."
            )

            competitors = [line.strip() for line in response_text.splitlines() if line.strip()]
            logger.debug("Parsed competitors: %s", competitors)
            return competitors[:5]  # Limit to top 5

        except Exception as e:
            logger.error(f"Competitor search error: {str(e)}")
            return []

    async def _search_market_data(self, sector: str) -> Dict[str, str]:
        """Search for market size and growth data"""
        try:
            queries = [
                f"{sector} market size TAM SAM",
                f"{sector} industry growth rate CAGR",
                f"{sector} market trends 2024 2025"
            ]

            all_results = []
            for query in queries:
                results = await self._perform_search(query, num_results=3, source="market")
                all_results.extend(results)

            if not all_results:
                return {}

            logger.debug("Market data search results: %s", all_results)
            # Extract market statistics
            combined_text = "".join([f"{r['title']}: {r['snippet']}" for r in all_results])
            response_text = self.summarizer.generate_text(
                f"""Extract market statistics for the {sector} sector from the following information:
                {combined_text}

                Return as JSON with keys: TAM (Total Addressable Market), SAM (Serviceable Addressable Market), CAGR (Compound Annual Growth Rate), key_trends.
                If specific data is not available, use "Not specified".
                """
            )

            try:
                import json
                logger.debug("Market data summary: %s", response_text)
                return json.loads(response_text.strip())
            except:
                return {"summary": response_text.strip()}

        except Exception as e:
            logger.error(f"Market data search error: {str(e)}")
            return {}

    async def _search_news(self, company_name: str, founder_name: List[str]) -> List[str]:
        """Search for recent news and updates"""
        try:
            founder_combined = ", ".join(founder_name)
            queries = [
                f"{company_name} funding investment news",
                f"{company_name} partnership launch news",
                f"{founder_combined} {company_name} announcement"
            ]

            news_items = []
            for query in queries:
                results = await self._perform_search(query, num_results=2, source="news")
                for result in results:
                    news_items.append(f"{result['title']}: {result['snippet']}")
            logger.debug("News items: %s", news_items)
            return news_items[:5]  # Limit to top 5 news items

        except Exception as e:
            logger.error(f"News search error: {str(e)}")
            return []

    async def _resolve_logo_companies(self, logos: Sequence[str]) -> List[Dict[str, str]]:
        """Resolve detected logo text to company names via search."""

        resolved: List[Dict[str, str]] = []
        seen_names = set()

        for logo in logos:
            results = await self._perform_search(
                f"{logo} company logo",
                num_results=3,
                source="logos",
            )
            entry = self._build_logo_entry(logo, results)
            if not entry:
                continue
            name_key = entry.get("company_name", "").lower()
            if name_key and name_key not in seen_names:
                seen_names.add(name_key)
                resolved.append(entry)

        return resolved

    @staticmethod
    def _build_logo_entry(
        logo: str,
        results: Sequence[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        """Build structured mapping for a logo search result."""

        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            candidate = PublicDataGatherer._select_company_name(logo, title, snippet)
            if candidate:
                return {
                    "logo_text": logo,
                    "company_name": candidate,
                    "source": result.get("link", ""),
                }

        fallback = PublicDataGatherer._select_company_name(logo, "", "")
        if not fallback:
            return None

        return {
            "logo_text": logo,
            "company_name": fallback,
            "source": "",
        }

    @staticmethod
    def _clean_company_title(title: str) -> str:
        """Normalise search result titles into company names."""

        if not title:
            return ""

        cleaned = title.strip()
        separators = [" - ", " | ", " · "]
        for sep in separators:
            if sep in cleaned:
                cleaned = cleaned.split(sep)[0]

        cleaned = re.sub(r"(?i)official site", "", cleaned)
        cleaned = re.sub(r"(?i)home page", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)

        if cleaned:
            return cleaned.strip()

        return title.strip()

    @staticmethod
    def _select_company_name(logo: str, title: str, snippet: str) -> str:
        """Choose the most plausible company name from search artefacts."""

        def _normalise_candidate(raw: str) -> str:
            return re.sub(r"[\s\-–:]+$", "", raw.strip())

        corp_keywords = (
            "company",
            "inc",
            "inc.",
            "corporation",
            "corp",
            "llc",
            "ltd",
            "group",
            "partners",
            "holdings",
            "technologies",
            "labs",
            "solutions",
        )

        banned_single_tokens = {
            "company",
            "inc",
            "inc.",
            "corporation",
            "corp",
            "llc",
            "ltd",
            "group",
            "partners",
            "holdings",
            "solutions",
        }

        def _score_candidate(candidate: str) -> Tuple[int, int]:
            lowered = candidate.lower()
            score = 0
            if any(keyword in lowered for keyword in corp_keywords):
                score += 3
            if " " in candidate:
                score += 2
            if "&" in candidate:
                score += 1
            if lowered == logo.lower():
                score -= 2
            return score, len(candidate)

        candidates: List[Tuple[str, Tuple[int, int]]] = []

        title_candidate = PublicDataGatherer._clean_company_title(title)
        if title_candidate:
            normalised = _normalise_candidate(title_candidate)
            lowered_title = normalised.lower()
            if len(normalised) >= 3 and lowered_title not in {"logo", "official site"}:
                if "logo" in lowered_title or any(ext in lowered_title for ext in ("png", "svg", "jpg")):
                    if not any(keyword in lowered_title for keyword in corp_keywords):
                        normalised = ""
                if normalised:
                    candidates.append((normalised, _score_candidate(normalised)))

        if snippet:
            pattern = re.compile(r"([A-Z][\w']+(?:\s+(?:&\s+)?[A-Z][\w']+){0,4})")
            for match in pattern.findall(snippet):
                normalised = _normalise_candidate(match)
                if len(normalised) < 3:
                    continue
                lowered = normalised.lower()
                if lowered in {"logo", "logos"}:
                    continue
                if len(normalised.split()) == 1 and lowered in banned_single_tokens:
                    continue
                if not any(keyword in lowered for keyword in corp_keywords) and "&" not in normalised and " " not in normalised:
                    continue
                candidates.append((normalised, _score_candidate(normalised)))

        if not candidates:
            trimmed_logo = _normalise_candidate(logo)
            if len(trimmed_logo) >= 4:
                candidates.append((trimmed_logo, _score_candidate(trimmed_logo)))

        if not candidates:
            return ""

        best = max(candidates, key=lambda item: item[1])
        # Filter out very short fallbacks (e.g., "B&")
        if len(best[0]) < 3 or best[0].lower() == logo.lower() and len(best[0]) < 4:
            return ""
        return best[0]

    async def _perform_search(
        self,
        query: str,
        *,
        num_results: int = 5,
        source: str = "general",
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._vector_query,
            query=query,
            source=source,
            k=num_results,
        )
#                 'news': results[3] if not isinstance(results[3], Exception) else []
#             }
#             return data
#         except Exception as e:
#             logger.error(f"Public data gathering error: {str(e)}")
#             return {}

#     # ===== Your search functions remain largely the same, just call new _perform_search =====
#     async def _search_founder_profile(self, founder_name: List[str]) -> str:
#         try:
#             founder_combined = ", ".join(founder_name)
#             queries = [
#                 f"{founder_combined} background experience",
#                 f"{founder_combined} LinkedIn profile career",
#                 f"{founder_combined} founder entrepreneur"
#             ]
#             all_results = []
#             for query in queries:
#                 results = await self._perform_search(query, num_results=3)
#                 all_results.extend(results)

#             if all_results:
#                 combined_text = "".join([f"{r['title']}: {r['snippet']}" for r in all_results])
#                 summary = self.summarizer.model.generate_content(
#                     f"Summarize the professional background of {founder_combined} based on {combined_text}"
#                 )
#                 return summary.text.strip()
#             return "No public information found"
#         except Exception as e:
#             logger.error(f"Founder search error: {str(e)}")
#             return "Error gathering founder information"

#     async def _search_competitors(self, company_name: str, sector: str) -> List[str]:
#         try:
#             query = f"{sector} companies competitors startups"
#             results = await self._perform_search(query, num_results=5)
#             if not results:
#                 return []
#             combined_text = "".join([f"{r['title']}: {r['snippet']}" for r in results])
#             response = self.summarizer.model.generate_content(
#                 f"Extract a list of company names that are competitors to {company_name} in the {sector} sector from: {combined_text}. Return only company names, one per line."
#             )
#             competitors = [line.strip() for line in response.text.splitlines() if line.strip()]
#             return competitors[:5]
#         except Exception as e:
#             logger.error(f"Competitor search error: {str(e)}")
#             return []

#     async def _search_market_data(self, sector: str) -> Dict[str, str]:
#         try:
#             queries = [
#                 f"{sector} market size TAM SAM",
#                 f"{sector} industry growth rate CAGR",
#                 f"{sector} market trends 2024 2025"
#             ]
#             all_results = []
#             for query in queries:
#                 results = await self._perform_search(query, num_results=3)
#                 all_results.extend(results)

#             if not all_results:
#                 return {}
#             combined_text = "".join([f"{r['title']}: {r['snippet']}" for r in all_results])
#             response = self.summarizer.model.generate_content(
#                 f"""Extract market statistics for the {sector} sector from the following information:
#                 {combined_text}
#                 Return as JSON with keys: TAM, SAM, CAGR, key_trends.
#                 If specific data is not available, use "Not specified"."""
#             )
#             import json
#             try:
#                 return json.loads(response.text.strip())
#             except:
#                 return {"summary": response.text.strip()}
#         except Exception as e:
#             logger.error(f"Market data search error: {str(e)}")
#             return {}

#     async def _search_news(self, company_name: str, founder_name: List[str]) -> List[str]:
#         try:
#             founder_combined = ", ".join(founder_name)
#             queries = [
#                 f"{company_name} funding investment news",
#                 f"{company_name} partnership launch news",
#                 f"{founder_combined} {company_name} announcement"
#             ]
#             news_items = []
#             for query in queries:
#                 results = await self._perform_search(query, num_results=2)
#                 for result in results:
#                     news_items.append(f"{result['title']}: {result['snippet']}")
#             return news_items[:5]
#         except Exception as e:
#             logger.error(f"News search error: {str(e)}")
#             return []
