from types import SimpleNamespace

import pytest

from utils.ocr_utils import (
    PAGE_LIMIT,
    DocumentAIPageLimitError,
    DocumentAIProcessingError,
    calculate_page_chunks,
    extract_text_from_pdf_docai,
)


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


class _FakeDocAIClient:
    def __init__(self, *, raise_page_limit: bool = False, message: str = "chunk text"):
        self.calls = []
        self.raise_page_limit = raise_page_limit
        self.response_text = message
        self._path = "projects/test/locations/us/processors/test"

    def processor_path(self, project: str, location: str, processor: str) -> str:  # pragma: no cover - simple passthrough
        return self._path

    def process_document(self, request):
        self.calls.append(request)
        if self.raise_page_limit:
            raise RuntimeError("PAGE_LIMIT_EXCEEDED: Document pages exceed limit")
        return SimpleNamespace(document=SimpleNamespace(text=self.response_text))


def test_extract_text_returns_text_on_success():
    client = _FakeDocAIClient()

    text = extract_text_from_pdf_docai(
        gcs_uri="gs://bucket/sample.pdf",
        project_id="p",
        location="loc",
        processor_id="proc",
        client=client,
        processor_resource=client.processor_path("p", "loc", "proc"),
    )

    assert text == "chunk text"
    assert len(client.calls) == 1


def test_extract_text_raises_for_page_limit():
    client = _FakeDocAIClient(raise_page_limit=True)

    with pytest.raises(DocumentAIPageLimitError):
        extract_text_from_pdf_docai(
            gcs_uri="gs://bucket/sample.pdf",
            project_id="p",
            location="loc",
            processor_id="proc",
            client=client,
            processor_resource=client.processor_path("p", "loc", "proc"),
        )


def test_extract_text_raises_for_generic_failures():
    class _AlwaysFailClient(_FakeDocAIClient):
        def process_document(self, request):  # pragma: no cover - deterministic failure
            raise RuntimeError("unexpected error")

    client = _AlwaysFailClient()

    with pytest.raises(DocumentAIProcessingError):
        extract_text_from_pdf_docai(
            gcs_uri="gs://bucket/sample.pdf",
            project_id="p",
            location="loc",
            processor_id="proc",
            client=client,
            processor_resource=client.processor_path("p", "loc", "proc"),
        )
