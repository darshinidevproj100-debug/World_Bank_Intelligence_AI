"""Tests for the verified World Bank Documents & Reports metadata API shape."""
import pytest

from src.data.documents_reports import fetch_document_metadata


class FakeResponse:
    def __init__(self, payload, url):
        self.payload = payload
        self.url = url

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, dict(params), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def api_document(document_id, title):
    return {
        "id": document_id,
        "entityids": {"entityid": f"WB-{document_id}"},
        "display_title": title,
        "docdt": "2024-01-15T00:00:00Z",
        "abstracts": {"cdata!": f"Evidence about {title.lower()} policy."},
        "url": f"https://documents.worldbank.org/record?docid={document_id}",
        "pdfurl": f"https://documents.worldbank.org/{document_id}.pdf",
    }


def test_document_search_paginates_and_preserves_official_metadata():
    session = FakeSession([
        FakeResponse({
            "total": 3,
            "documents": {
                "D1": api_document("1", "Education Policy"),
                "D2": api_document("2", "Education Recovery"),
                "facets": {},
            },
        }, "https://search.worldbank.org/api/v3/wds?os=0"),
        FakeResponse({
            "total": 3,
            "documents": {"D3": api_document("3", "School Resilience")},
        }, "https://search.worldbank.org/api/v3/wds?os=2"),
    ])

    documents = fetch_document_metadata("education policy", limit=3, session=session)

    assert [document["document_key"] for document in documents] == ["D1", "D2", "D3"]
    assert documents[0]["abstract"] == "Evidence about education policy policy."
    assert documents[0]["entity_id"] == "WB-1"
    assert documents[0]["source_url"].endswith("docid=1")
    assert documents[0]["pdf_url"].endswith("1.pdf")
    assert documents[0]["ingested_at_utc"].endswith("+00:00")
    assert [call[1]["os"] for call in session.calls] == [0, 2]
    assert all(call[1]["qterm"] == "education policy" for call in session.calls)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": ""}, "query"),
        ({"limit": 0}, "limit"),
        ({"offset": -1}, "offset"),
        ({"timeout": float("inf")}, "timeout"),
        ({"max_retries": 6}, "max_retries"),
    ],
)
def test_document_search_validates_inputs(kwargs, message):
    query = kwargs.pop("query", "education")
    with pytest.raises(ValueError, match=message):
        fetch_document_metadata(query, **kwargs)
