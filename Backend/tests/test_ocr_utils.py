from utils.ocr_utils import PAGE_LIMIT, calculate_page_chunks


def test_calculate_page_chunks_for_empty_document():
    assert calculate_page_chunks(0) == []


def test_calculate_page_chunks_under_limit():
    assert calculate_page_chunks(5) == [(0, 5)]


def test_calculate_page_chunks_exact_limit():
    assert calculate_page_chunks(PAGE_LIMIT) == [(0, PAGE_LIMIT)]


def test_calculate_page_chunks_over_limit():
    expected = [(0, PAGE_LIMIT), (PAGE_LIMIT, PAGE_LIMIT + 1)]
    assert calculate_page_chunks(PAGE_LIMIT + 1) == expected


def test_calculate_page_chunks_multiple_overflows():
    total_pages = (PAGE_LIMIT * 2) + 6
    expected = [
        (0, PAGE_LIMIT),
        (PAGE_LIMIT, PAGE_LIMIT * 2),
        (PAGE_LIMIT * 2, total_pages),
    ]
    assert calculate_page_chunks(total_pages) == expected


def test_calculate_page_chunks_custom_limit():
    assert calculate_page_chunks(10, page_limit=4) == [(0, 4), (4, 8), (8, 10)]


def test_calculate_page_chunks_invalid_inputs():
    try:
        calculate_page_chunks(-1)
    except ValueError as exc:
        assert "total_pages" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("calculate_page_chunks should reject negative totals")

    try:
        calculate_page_chunks(1, page_limit=0)
    except ValueError as exc:
        assert "page_limit" in str(exc)
    else:  # pragma: no cover - sanity guard
        raise AssertionError("calculate_page_chunks should reject non-positive limits")
