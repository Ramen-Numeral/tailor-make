# TailorMake

TailorMake is an evidence-grounded resume tailoring agent built to make its
work visible. It streams a
traceable account of what it found, what it changed, why it changed it, and
which job requirements the candidate's resume does not support.

The application parses a resume and job listing, maps requirements to
candidate evidence, creates a positioning plan, rewrites within explicit
budgets, validates the result, and shows the complete decision trail in an
interactive React interface. The final resume can be reviewed, edited, and
exported as a PDF.

## Why TailorMake

Resume tailoring tools often hide their reasoning and make it difficult to
tell whether a polished claim is actually supported. TailorMake treats
transparency and factual integrity as part of the product:

- **Evidence-backed matching:** each job requirement is classified as
  supported, partially supported, or unsupported and linked to candidate
  evidence.
- **Observable agent loop:** planning, drafting, evaluation, revision, and
  page-fitting events are streamed to the interface as they happen.
- **Traceable decisions:** events include observations, actions, decision
  reasons, affected sections, and rewrite attempts.
- **Bounded revision:** configurable rewrite and page-trim budgets prevent an
  open-ended agent loop.
- **Factual safeguards:** unsupported requirements are surfaced as gaps rather
  than silently invented as candidate experience.
- **Human feedback:** when enabled, the agent can pause on evidence gaps and
  ask the candidate for omitted facts before continuing.
- **Visible outcomes:** before-and-after differences, requirement coverage,
  match scores, bullet-quality checks, recruiter-oriented evaluation, and
  page-fit actions remain available for review.

The trace is an application-level record of inputs, observations, decisions,
and actions. It is designed for inspection and accountability; it does not
expose a model provider's private chain-of-thought.

## How the agent loop works

1. Parse the job listing into explicit requirements.
2. Find supporting resume evidence using lexical matching, optional sentence
   embeddings, and LLM adjudication.
3. Mark weak or missing evidence instead of fabricating qualifications.
4. Build a positioning brief and section-level writing plan.
5. Select and rewrite relevant content.
6. Validate factual consistency, section constraints, and writing quality.
7. Accept, restore, or revise sections within the configured rewrite budget.
8. Evaluate the resume from a recruiter perspective.
9. Fit the result to the requested page limit and record every trim.
10. Return the resume, diffs, scores, actions, warnings, and full event trace.

## Application components

- FastAPI backend with server-sent progress events
- React and Vite interface for feedback, diffs, edits, and export
- evidence-aware keyword matching with BM25 and optional sentence embeddings
- configurable AI-content detection and resume validation
- HTML and PDF rendering
- automated unit, integration, feature, edge-case, and system tests
- optional data preparation and model-training pipelines

## Requirements

- Python 3.11 or newer
- Node.js 18 or newer and npm
- a [Groq API key](https://console.groq.com/keys)
- system libraries required by WeasyPrint for PDF generation

Use `DEVICE=cpu` unless the machine has a working CUDA environment.

## Quick start

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

For the standard application demo, use:

```dotenv
GROQ_API_KEY=your-key-here
APP_ENV=development
DEVICE=cpu
AI_DETECTION_ENABLED=false
KEYWORD_EMBEDDINGS_ENABLED=true
```

Then install the frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

### Important: AI detection is optional

You do **not** need to run the training notebooks or pipelines to use the
resume-tailoring agent.

The AI-content detector expects locally trained artifacts such as the
CatBoost, DistilBERT, TF-IDF, and rubric-regression models beneath `models/`.
Those artifacts are not included in the repository. If you have not trained
or supplied them, keep this setting in `.env`:

```dotenv
AI_DETECTION_ENABLED=false
```

Set it to `true` only when the expected detector artifacts are available:

```dotenv
AI_DETECTION_ENABLED=true
```

`KEYWORD_EMBEDDINGS_ENABLED` is separate from AI detection. When enabled, it
uses a pinned sentence-transformer model to improve evidence matching; its
first use may download model files. For the lightest possible setup, disable
that feature too:

```dotenv
AI_DETECTION_ENABLED=false
KEYWORD_EMBEDDINGS_ENABLED=false
```

The `ml_pipelines/` directory is for preparing data and training or evaluating
optional local models. It is not part of the normal web-app startup path.

## Run locally

Start the API from the repository root:

```bash
.venv/bin/uvicorn app.web.api:app --reload
```

In another terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. The API is available at
<http://localhost:8000>, with interactive documentation at
<http://localhost:8000/docs>.

## Production-like local run

Build the frontend and serve the complete application from FastAPI:

```bash
cd frontend
npm run build
cd ..
.venv/bin/uvicorn app.web.api:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

## Command-line demo

`app.py` contains a sample job listing and runs the tailoring pipeline against
the candidate profile in `config/resume/`. It prints agent trace events and
the evidence-coverage plan while the run proceeds:

```bash
.venv/bin/python app.py
```

Generated HTML and PDF files are written beneath `output/`.

## Tests

Install pytest if it is not already available, then run:

```bash
python -m pip install pytest
python -m pytest
```

To run only the web API tests:

```bash
python -m pytest tests/features/test_web_api.py
```

## Project structure

```text
app/
  features/          Agent loop, evidence planning, writing, and validation
  infrastructure/    LLM, model, cache, and logging adapters
  resume_schema/     Core resume data models
  web/               FastAPI application and streamed trace API
config/               Runtime, model, I/O, and candidate configuration
frontend/             React interface
ml_pipelines/         Optional data preparation and model training
tests/                Automated test suite
```

## Configuration

Runtime configuration is loaded from `.env`. The principal feature flags are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | unset | Authenticates Groq-backed parsing, writing, and evaluation |
| `DEVICE` | `cpu` | Selects the local inference device |
| `AI_DETECTION_ENABLED` | `true` | Runs the optional locally trained AI-content detector |
| `KEYWORD_EMBEDDINGS_ENABLED` | `true` | Adds sentence-embedding evidence retrieval |
| `LOG_LEVEL` | `INFO` | Controls application logging verbosity |

Although AI detection defaults to enabled in the application configuration,
users without the trained artifacts should explicitly set
`AI_DETECTION_ENABLED=false`.

See `.env.example` for the core settings. Paths and filenames are defined in
`config/settings.py` and `config/io.py`; LLM routes are defined in
`config/llm.py`.

Do not commit `.env` or API keys.
