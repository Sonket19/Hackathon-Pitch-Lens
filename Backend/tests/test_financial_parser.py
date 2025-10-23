from utils.financial_parser import (
    backfill_financials_section,
    extract_financial_highlights,
)


def _sample_extracted_text() -> dict:
    return {
        "pitch_deck": {
            "raw": {
                "1": "ARR: $1.2M\nMonthly Burn: $80k/mo\nRunway: 12 months",
                "2": "MRR - $100k\nFunding ask: INR 5 Cr",
            }
        }
    }


def test_extract_financial_highlights_recovers_metrics():
    highlights = extract_financial_highlights(_sample_extracted_text())

    assert highlights["arr"].upper() == "$1.2M"
    assert highlights["mrr"].upper() == "$100K"
    assert highlights["burn"].lower().startswith("$80k")
    assert highlights["runway"] == "12 months"
    assert highlights["funding_ask"].upper().startswith("INR 5")


def test_backfill_financials_section_updates_missing_fields():
    memo = {
        "financials": {
            "srr_mrr": {
                "current_booked_arr": "Not available",
                "current_mrr": "Not available",
            },
            "burn_and_runway": {
                "funding_ask": "Not available",
                "stated_runway": "",
                "implied_net_burn": None,
            },
        }
    }

    highlights = extract_financial_highlights(_sample_extracted_text())
    backfill_financials_section(memo, highlights)

    financials = memo["financials"]
    arr_section = financials.get("arr_mrr", {})
    burn_section = financials.get("burn_and_runway", {})

    assert arr_section["current_booked_arr"].upper() == "$1.2M"
    assert arr_section["current_mrr"].upper() == "$100K"
    assert burn_section["funding_ask"].upper().startswith("INR 5")
    assert burn_section["stated_runway"] == "12 months"
    assert burn_section["implied_net_burn"].lower().startswith("$80k")
    assert "srr_mrr" not in financials


def test_backfill_preserves_existing_values():
    memo = {
        "financials": {
            "arr_mrr": {
                "current_booked_arr": "$2.4M",
                "current_mrr": "$200k",
            },
            "burn_and_runway": {
                "funding_ask": "USD 3M",
                "stated_runway": "18 months",
                "implied_net_burn": "$150k/mo",
            },
        }
    }

    highlights = extract_financial_highlights(_sample_extracted_text())
    backfill_financials_section(memo, highlights)

    financials = memo["financials"]
    arr_section = financials.get("arr_mrr", {})
    burn_section = financials.get("burn_and_runway", {})

    assert arr_section["current_booked_arr"] == "$2.4M"
    assert arr_section["current_mrr"] == "$200k"
    assert burn_section["funding_ask"] == "USD 3M"
    assert burn_section["stated_runway"] == "18 months"
    assert burn_section["implied_net_burn"] == "$150k/mo"
