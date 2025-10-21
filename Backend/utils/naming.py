"""Helpers for presenting company metadata consistently."""
from __future__ import annotations

from typing import Optional


def build_company_display_name(
    company_name: Optional[str],
    product_name: Optional[str],
) -> str:
    """Return a combined display name that shows both company and product when distinct.

    The function keeps existing whitespace tidy and avoids duplicating values when the
    product name already appears inside the company name (case-insensitive).
    """

    company = (company_name or "").strip()
    product = (product_name or "").strip()

    if not company and not product:
        return ""

    if not product:
        return company

    if not company:
        return product

    normalized_company = company.lower()
    normalized_product = product.lower()

    if normalized_product == normalized_company:
        return company

    if normalized_product in normalized_company or normalized_company in normalized_product:
        return company

    return f"{company} — {product}"
