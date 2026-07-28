"""System-boundary tests using the real FastAPI app and run registry."""

from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.web import api as api_module
from app.web.api import RunRecord, StartRunRequest, app
from config.resume.candidate_profile import build_resume


def test_pdf_download_is_a_real_parseable_document() -> None:
    resume = build_resume()

    response = TestClient(app).post(
        f"/api/runs/{uuid4()}/pdf",
        json={
            "confirmed": True,
            "resume": resume.model_dump(mode="json"),
        },
    )

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-")
    assert len(PdfReader(BytesIO(response.content)).pages) >= 1
    assert "attachment;" in response.headers["content-disposition"]


def test_human_input_endpoint_transitions_a_real_run_record() -> None:
    record = RunRecord(
        StartRunRequest(
            resume=build_resume(),
            job_listing="Backend engineer with Python experience.",
            human_in_the_loop=True,
        )
    )
    record.running = True
    record.awaiting_human_input = True
    with api_module._runs_lock:
        api_module._runs[record.id] = record

    try:
        response = TestClient(app).post(
            f"/api/runs/{record.id}/human-input",
            json={"notes": "Used PostgreSQL for weekly product reporting."},
        )

        assert response.status_code == 200
        assert response.json() == {"status": "accepted"}
        assert record.awaiting_human_input is False
        assert record.human_input_event.is_set()
        assert record.human_input_notes == (
            "Used PostgreSQL for weekly product reporting."
        )
    finally:
        with api_module._runs_lock:
            api_module._runs.pop(record.id, None)


def test_human_input_rejects_stale_and_oversized_submissions() -> None:
    record = RunRecord(
        StartRunRequest(
            resume=build_resume(),
            job_listing="Backend engineer.",
        )
    )
    with api_module._runs_lock:
        api_module._runs[record.id] = record

    try:
        client = TestClient(app)
        stale = client.post(
            f"/api/runs/{record.id}/human-input",
            json={"notes": "Too late"},
        )
        oversized = client.post(
            f"/api/runs/{record.id}/human-input",
            json={"notes": "x" * 5001},
        )

        assert stale.status_code == 409
        assert "not waiting" in stale.json()["detail"]
        assert oversized.status_code == 422
    finally:
        with api_module._runs_lock:
            api_module._runs.pop(record.id, None)


def test_api_validation_rejects_invalid_run_boundaries() -> None:
    client = TestClient(app)
    resume = build_resume().model_dump(mode="json")

    empty_listing = client.post(
        "/api/runs",
        json={"resume": resume, "job_listing": ""},
    )
    excessive_pages = client.post(
        "/api/runs",
        json={
            "resume": resume,
            "job_listing": "Engineer",
            "maximum_pages": 99,
        },
    )
    unconfirmed_pdf = client.post(
        f"/api/runs/{uuid4()}/pdf",
        json={"confirmed": False, "resume": resume},
    )

    assert empty_listing.status_code == 422
    assert excessive_pages.status_code == 422
    assert unconfirmed_pdf.status_code == 400
