# TailorMake

TailorMake is an AI-assisted resume tailoring application. It parses a
candidate's resume, analyzes a job listing, finds supported keyword evidence,
rewrites and validates the resume, shows the changes in an interactive React
interface, and exports the result as a PDF.

The project includes:

- a FastAPI backend with streamed run progress
- a React and Vite frontend
- evidence-aware keyword matching with BM25 and sentence embeddings
- configurable AI-content detection and resume validation
- HTML and PDF rendering
- an automated Python test suite

## Requirements

- Python 3.11 or newer
- Node.js 18 or newer and npm
- a [Groq API key](https://console.groq.com/keys)
- system libraries required by WeasyPrint for PDF generation

The first use of the embedding and transformer features may download model
files. Set `DEVICE=cpu` unless your environment has a working CUDA setup.

## Setup

From the repository root, create a virtual environment and install the Python
dependencies:

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

Add your Groq key to `.env` and adjust the runtime values for your machine:

```dotenv
GROQ_API_KEY=your-key-here
APP_ENV=development
DEVICE=cpu
AI_DETECTION_ENABLED=true
```

Install the frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

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
<http://localhost:8000>, with interactive API documentation at
<http://localhost:8000/docs>.

## Production-like local run

Build the frontend and serve it from FastAPI:

```bash
cd frontend
npm run build
cd ..
.venv/bin/uvicorn app.web.api:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

## Command-line demo

`app.py` contains a sample job listing and runs the complete tailoring
pipeline against the candidate profile in `config/resume/`:

```bash
.venv/bin/python app.py
```

Generated HTML and PDF files are written beneath `output/`.

## Tests

Install pytest if it is not already available in the environment, then run:

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
  features/          Resume parsing, tailoring, validation, and rendering
  infrastructure/    LLM, model, cache, and logging adapters
  resume_schema/     Core resume data models
  web/               FastAPI application
config/               Runtime, model, I/O, and candidate configuration
frontend/             React and Vite user interface
ml_pipelines/         Data preparation and model-training utilities
tests/                Unit, integration, feature, edge-case, and system tests
```

## Configuration

Runtime configuration is loaded from `.env`. See `.env.example` for the
available core settings. Application paths and filenames are defined in
`config/settings.py` and `config/io.py`; model routes are defined in
`config/llm.py`.

Do not commit `.env` or API keys.
