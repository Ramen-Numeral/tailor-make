from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from app.web import api as api_module
from app.web.api import HumanInputRequest, UpdateRunResumeRequest
from app.infrastructure.llm.usage import meter_llm_usage, record_llm_call
from app.web.api import app
from config.resume.candidate_profile import build_resume


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pdf_upload_rejects_wrong_media_type() -> None:
    response = TestClient(app).post(
        "/api/profile/parse-pdf",
        files={"file": ("resume.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Upload must be a PDF."


def test_usage_meter_counts_calls_and_estimates_tokens() -> None:
    with meter_llm_usage() as usage:
        record_llm_call(8, "12345678")
        record_llm_call(4, "1234")

    assert usage.calls == 2
    assert usage.prompt_tokens_estimated == 3
    assert usage.completion_tokens_estimated == 3
    assert usage.total_tokens_estimated == 6


def test_candidate_edits_update_export_copy_not_source(monkeypatch) -> None:
    source = build_resume()
    edited = source.model_copy(
        update={
            "candidate": source.candidate.model_copy(
                update={"target_title": "Edited export title"}
            )
        },
        deep=True,
    )
    state = SimpleNamespace(
        selected_resume=source,
        current_resume=source,
        diffs=[],
    )
    result = SimpleNamespace(
        state=state,
        resume=source,
        diffs=[],
        model_dump=lambda **kwargs: {"candidate": "edited"},
    )
    record = SimpleNamespace(
        running=False,
        result=result,
        source=source,
        current=source,
    )
    monkeypatch.setattr(api_module, "_get_run", lambda run_id: record)

    api_module.update_run_resume(
        source.work_experience.items[0].id,
        UpdateRunResumeRequest(resume=edited),
    )

    assert record.current.candidate.target_title == "Edited export title"
    assert record.result.resume == edited
    assert record.source.candidate.target_title != "Edited export title"


def test_pdf_export_survives_missing_in_memory_run(monkeypatch) -> None:
    resume = build_resume()
    monkeypatch.setattr(
        api_module,
        "render_pdf_bytes",
        lambda html: b"%PDF-reloaded-export",
    )

    response = TestClient(app).post(
        f"/api/runs/{uuid4()}/pdf",
        json={
            "confirmed": True,
            "resume": resume.model_dump(mode="json"),
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-reloaded-export"


def test_pdf_export_without_run_or_resume_has_clear_error() -> None:
    response = TestClient(app).post(
        f"/api/runs/{uuid4()}/pdf",
        json={"confirmed": True},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "No completed resume is available."


def test_human_input_resumes_waiting_run(monkeypatch) -> None:
    from threading import Event

    record = SimpleNamespace(
        awaiting_human_input=True,
        human_input_notes=None,
        human_input_event=Event(),
    )
    monkeypatch.setattr(api_module, "_get_run", lambda run_id: record)

    response = api_module.provide_human_input(
        uuid4(),
        HumanInputRequest(notes="Used PostgreSQL in weekly reporting."),
    )

    assert response == {"status": "accepted"}
    assert record.human_input_notes == "Used PostgreSQL in weekly reporting."
    assert record.human_input_event.is_set()
