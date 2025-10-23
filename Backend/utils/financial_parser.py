"""Utilities for extracting and backfilling financial metrics from OCR text."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


VALUE_PATTERN = re.compile(
    r"""
    (?P<prefix>(?:usd|eur|inr|sgd|aud|cad|gbp|rs|idr|nzd|zar|chf)\s*)?
    (?P<currency>[\$€£₹])?\s*
    (?P<number>\d[\d,]*(?:\.\d+)?)
    (?P<suffix>\s*(?:k|m|b|mn|bn|mm|million|billion|trillion|crore|crores|cr|lakh|lakhs|l|bn)\.?)?
    (?P<trailing>\s*(?:usd|eur|inr|sgd|aud|cad|gbp))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

RUNWAY_PATTERN = re.compile(r"\b\d+\s*(?:months?|mos?|month|mo|yrs?|years?)\b", re.IGNORECASE)

PLACEHOLDER_VALUES = {
    "",
    "n/a",
    "na",
    "not available",
    "not applicable",
    "unknown",
    "--",
    "-",
}


def _normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_value_from_line(line: str) -> Optional[str]:
    match = VALUE_PATTERN.search(line)
    if not match:
        return None

    prefix = (match.group("prefix") or "").strip()
    currency = (match.group("currency") or "").strip()
    number = match.group("number") or ""
    suffix = (match.group("suffix") or "").strip()
    trailing = (match.group("trailing") or "").strip()

    components: List[str] = []
    if prefix:
        components.append(prefix.upper())

    core_value = f"{currency}{number}" if currency else number
    if suffix:
        cleaned_suffix = suffix.rstrip(".").strip()
        if cleaned_suffix:
            compact_suffixes = {"k", "m", "b", "mn", "bn", "mm"}
            if cleaned_suffix.lower() in compact_suffixes:
                core_value = f"{core_value}{cleaned_suffix.upper()}" if core_value else cleaned_suffix.upper()
            else:
                core_value = f"{core_value} {cleaned_suffix}" if core_value else cleaned_suffix
    components.append(core_value)

    if trailing:
        components.append(trailing.upper())

    value = " ".join(comp for comp in components if comp)
    return _normalise_whitespace(value)


def _extract_runway_from_line(line: str) -> Optional[str]:
    match = RUNWAY_PATTERN.search(line)
    if not match:
        return None
    return _normalise_whitespace(match.group(0))


def _candidate_from_line(line: str, *, keywords: Iterable[str]) -> Optional[str]:
    cleaned = line.strip("•*-\t ")
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if not any(keyword in lowered for keyword in keywords):
        return None

    # Prefer the segment after the keyword or punctuation.
    for keyword in keywords:
        idx = lowered.find(keyword)
        if idx != -1:
            after = cleaned[idx + len(keyword):]
            candidate = after.strip(" :–-—")
            if candidate:
                return candidate

    # Fall back to the portion after colon/dash if present.
    for token in (":", "-", "–", "—"):
        if token in cleaned:
            candidate = cleaned.split(token, 1)[1].strip()
            if candidate:
                return candidate

    return cleaned


def _extract_metric(text: str, *, keywords: Iterable[str], runway: bool = False) -> Optional[str]:
    for line in text.splitlines():
        candidate_fragment = _candidate_from_line(line, keywords=keywords)
        if not candidate_fragment:
            continue

        if runway:
            runway_value = _extract_runway_from_line(candidate_fragment)
            if runway_value:
                return runway_value

        value = _extract_value_from_line(candidate_fragment)
        if value:
            # Preserve monthly cadence hints when present.
            lowered_line = candidate_fragment.lower()
            if any(hint in lowered_line for hint in ("/mo", "per month", "monthly")):
                if "per month" in lowered_line:
                    value = f"{value} per month"
                elif "monthly" in lowered_line and not value.lower().endswith("monthly"):
                    value = f"{value} monthly"
                elif "/mo" in lowered_line and not value.lower().endswith("/mo"):
                    value = f"{value}/mo"
            return value

        if runway:
            runway_value = _extract_runway_from_line(candidate_fragment)
            if runway_value:
                return runway_value

        candidate = _normalise_whitespace(candidate_fragment)
        if candidate:
            return candidate

    return None


def extract_financial_highlights(extracted_text: Dict[str, Any]) -> Dict[str, str]:
    """Parse OCR text to recover key financial metrics when the model omits them."""

    if not isinstance(extracted_text, dict):
        return {}

    pitch_payload = extracted_text.get("pitch_deck")
    text_sources: List[str] = []

    if isinstance(pitch_payload, dict):
        raw_pages = pitch_payload.get("raw")
        if isinstance(raw_pages, dict):
            for _, page_text in sorted(raw_pages.items(), key=lambda item: item[0]):
                if isinstance(page_text, str) and page_text.strip():
                    text_sources.append(page_text)
        elif isinstance(raw_pages, list):
            for page_text in raw_pages:
                if isinstance(page_text, str) and page_text.strip():
                    text_sources.append(page_text)

    if not text_sources:
        return {}

    combined_text = "\n".join(text_sources)

    highlights: Dict[str, str] = {}

    arr_value = _extract_metric(combined_text, keywords=("arr", "annual recurring revenue"))
    if arr_value:
        highlights["arr"] = arr_value

    mrr_value = _extract_metric(combined_text, keywords=("mrr", "monthly recurring revenue"))
    if mrr_value:
        highlights["mrr"] = mrr_value

    burn_value = _extract_metric(
        combined_text,
        keywords=("burn", "burn rate", "net burn", "monthly burn"),
    )
    if burn_value:
        highlights["burn"] = burn_value

    runway_value = _extract_metric(
        combined_text,
        keywords=("runway", "cash runway"),
        runway=True,
    )
    if runway_value:
        highlights["runway"] = runway_value

    funding_value = _extract_metric(
        combined_text,
        keywords=("funding ask", "raise", "seeking", "amount raising"),
    )
    if funding_value:
        highlights["funding_ask"] = funding_value

    return highlights


def _needs_backfill(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return False
    if not isinstance(value, str):
        return False
    return value.strip().lower() in PLACEHOLDER_VALUES


def backfill_financials_section(memo: Dict[str, Any], highlights: Dict[str, str]) -> None:
    """Ensure memo financial keys align with the UI expectations and patch missing values."""

    if not isinstance(memo, dict) or not highlights:
        return

    financials = memo.get("financials")
    if not isinstance(financials, dict):
        financials = {}
        memo["financials"] = financials

    arr_section: Dict[str, Any] = {}
    if isinstance(financials.get("arr_mrr"), dict):
        arr_section = dict(financials["arr_mrr"])
    elif isinstance(financials.get("srr_mrr"), dict):
        arr_section = dict(financials["srr_mrr"])

    if highlights.get("arr") and _needs_backfill(arr_section.get("current_booked_arr")):
        arr_section["current_booked_arr"] = highlights["arr"]
    if highlights.get("mrr") and _needs_backfill(arr_section.get("current_mrr")):
        arr_section["current_mrr"] = highlights["mrr"]

    if arr_section:
        financials["arr_mrr"] = arr_section

    if "srr_mrr" in financials:
        financials.pop("srr_mrr", None)

    burn_section: Dict[str, Any] = {}
    if isinstance(financials.get("burn_and_runway"), dict):
        burn_section = dict(financials["burn_and_runway"])

    if highlights.get("funding_ask") and _needs_backfill(burn_section.get("funding_ask")):
        burn_section["funding_ask"] = highlights["funding_ask"]
    if highlights.get("runway") and _needs_backfill(burn_section.get("stated_runway")):
        burn_section["stated_runway"] = highlights["runway"]
    if highlights.get("burn") and _needs_backfill(burn_section.get("implied_net_burn")):
        burn_section["implied_net_burn"] = highlights["burn"]

    if burn_section:
        financials["burn_and_runway"] = burn_section

