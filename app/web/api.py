"""FastAPI facade for the interactive resume-tailoring demo."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from app.features.agent.orchestrator import tailor_resume_agent
from app.features.agent.schema import AgentBudget, TailoringRunResult
from app.features.renderer.renderer import render_html, render_pdf_bytes
from app.features.resume_parser import ResumePDFError, parse_resume_pdf
from app.features.resume_parser.pdf import DEFAULT_MAX_PDF_BYTES
from app.features.resume_diff.differ import build_resume_diffs
from app.infrastructure.llm.usage import meter_llm_usage
from app.infrastructure.llm.errors import LLMError
from app.resume_schema.resume_schema import Resume


class StartRunRequest(BaseModel):
    resume: Resume
    job_listing: str = Field(min_length=1)
    maximum_pages: int = Field(default=1, ge=1, le=3)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    include_summary: bool = True
    human_in_the_loop: bool = False


class EditRunRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=2000)


class UpdateRunResumeRequest(BaseModel):
    resume: Resume


class ConfirmPDFRequest(BaseModel):
    confirmed: bool
    resume: Resume | None = None


class HumanInputRequest(BaseModel):
    notes: str = Field(default="", max_length=5000)


class SPAStaticFiles(StaticFiles):
    """Serve built assets and fall back to index.html for client-side routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code != 404:
                raise
        else:
            if response.status_code != 404:
                return response

        is_api_path = path == "api" or path.startswith("api/")
        if scope["method"] not in {"GET", "HEAD"} or is_api_path:
            raise StarletteHTTPException(status_code=404)
        return await super().get_response("index.html", scope)


class RunRecord:
    def __init__(self, request: StartRunRequest) -> None:
        self.id = uuid4()
        self.source = request.resume
        self.current = request.resume
        self.job_listing = request.job_listing
        self.maximum_pages = request.maximum_pages
        self.budget = request.budget
        self.include_summary = request.include_summary
        self.human_in_the_loop = request.human_in_the_loop
        self.human_input_event = Event()
        self.human_input_notes: str | None = None
        self.awaiting_human_input = False
        self.events: Queue[dict[str, Any]] = Queue()
        self.result: TailoringRunResult | None = None
        self.running = False


app = FastAPI(title="TailorMake", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runs: dict[UUID, RunRecord] = {}
_runs_lock = Lock()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/profile/parse-pdf")
async def parse_profile_pdf(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload must be a PDF.")
    try:
        with meter_llm_usage() as usage:
            parsed = parse_resume_pdf(
                await file.read(DEFAULT_MAX_PDF_BYTES + 1),
                filename=file.filename,
            )
    except ResumePDFError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except LLMError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Resume text was extracted, but the parsing model is "
                "temporarily unavailable or rate-limited. Please retry shortly."
            ),
        ) from error
    return {
        **parsed.model_dump(mode="json"),
        "usage": usage.as_dict(),
    }


@app.post("/api/runs", status_code=202)
def start_run(request: StartRunRequest) -> dict[str, str]:
    record = RunRecord(request)
    with _runs_lock:
        _runs[record.id] = record
    _start_worker(record)
    return {"run_id": str(record.id)}


@app.get("/api/runs/{run_id}/events")
def stream_run(run_id: UUID) -> StreamingResponse:
    record = _get_run(run_id)

    def generate():
        while True:
            try:
                message = record.events.get(timeout=15)
            except Empty:
                yield ": keepalive\n\n"
                continue
            yield f"event: {message['type']}\n"
            yield f"data: {json.dumps(message['data'])}\n\n"
            if message["type"] in {"completed", "failed"}:
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/runs/{run_id}/edits", status_code=202)
def edit_run(run_id: UUID, request: EditRunRequest) -> dict[str, str]:
    record = _get_run(run_id)
    if record.running:
        raise HTTPException(status_code=409, detail="The run is still active.")
    _start_worker(record, instruction=request.instruction)
    return {"status": "started"}


@app.post("/api/runs/{run_id}/human-input")
def provide_human_input(
    run_id: UUID,
    request: HumanInputRequest,
) -> dict[str, str]:
    record = _get_run(run_id)
    if not record.awaiting_human_input:
        raise HTTPException(
            status_code=409,
            detail="This run is not waiting for candidate input.",
        )
    record.human_input_notes = request.notes.strip() or None
    record.awaiting_human_input = False
    record.human_input_event.set()
    return {"status": "accepted"}


@app.put("/api/runs/{run_id}/resume")
def update_run_resume(
    run_id: UUID,
    request: UpdateRunResumeRequest,
) -> dict[str, Any]:
    """Persist candidate edits to the tailored copy, never the source."""
    record = _get_run(run_id)
    if record.running:
        raise HTTPException(status_code=409, detail="The run is still active.")
    if record.result is None:
        raise HTTPException(status_code=409, detail="No completed resume is available.")

    diffs = build_resume_diffs(
        record.source,
        record.result.state.selected_resume,
        request.resume,
    )
    record.current = request.resume
    record.result.resume = request.resume
    record.result.diffs = diffs
    record.result.state.current_resume = request.resume
    record.result.state.diffs = diffs
    return {"result": record.result.model_dump(mode="json")}


@app.post("/api/runs/{run_id}/pdf")
def download_pdf(run_id: UUID, request: ConfirmPDFRequest) -> Response:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="PDF export was not confirmed.")
    record = _find_run(run_id)
    if record is not None and record.running:
        raise HTTPException(status_code=409, detail="The run is still active.")
    resume = request.resume or (
        record.current
        if record is not None and record.result is not None
        else None
    )
    if resume is None:
        raise HTTPException(
            status_code=409,
            detail="No completed resume is available.",
        )
    pdf = render_pdf_bytes(render_html(resume))
    safe_name = "-".join(resume.candidate.name.lower().split()) or "resume"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}-resume.pdf"'
        },
    )


def _get_run(run_id: UUID) -> RunRecord:
    record = _find_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return record


def _find_run(run_id: UUID) -> RunRecord | None:
    with _runs_lock:
        return _runs.get(run_id)


def _start_worker(record: RunRecord, instruction: str | None = None) -> None:
    record.running = True

    def work() -> None:
        try:
            def request_human_input(_coverage_plan) -> str | None:
                record.human_input_notes = None
                record.human_input_event.clear()
                record.awaiting_human_input = True
                record.human_input_event.wait()
                return record.human_input_notes

            with meter_llm_usage() as usage:
                result = tailor_resume_agent(
                    record.current,
                    record.job_listing,
                    special_instructions=instruction,
                    budget=record.budget,
                    maximum_pages=record.maximum_pages,
                    include_summary=record.include_summary,
                    trace_callback=lambda event: record.events.put(
                        {
                            "type": "trace",
                            "data": event.model_dump(mode="json"),
                        }
                    ),
                    human_input_callback=(
                        request_human_input
                        if record.human_in_the_loop and instruction is None
                        else None
                    ),
                )
            cumulative_diffs = build_resume_diffs(
                record.source,
                result.state.selected_resume,
                result.resume,
                page_trim_actions=result.page_check.trim_actions,
            )
            result.diffs = cumulative_diffs
            result.state.diffs = cumulative_diffs
            record.current = result.resume
            record.result = result
            record.events.put(
                {
                    "type": "completed",
                    "data": {
                        "result": result.model_dump(mode="json"),
                        "usage": usage.as_dict(),
                        "edit_instruction": instruction,
                    },
                }
            )
        except Exception as error:
            record.events.put(
                {
                    "type": "failed",
                    "data": {"message": str(error) or type(error).__name__},
                }
            )
        finally:
            record.running = False

    Thread(target=work, daemon=True, name=f"resume-run-{record.id}").start()


_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount(
        "/",
        SPAStaticFiles(directory=_frontend_dist, html=True),
        name="frontend",
    )
