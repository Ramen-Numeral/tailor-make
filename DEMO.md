# Resume Lab portfolio demo

## Development

Start the API from the repository root:

```bash
.venv/bin/uvicorn app.web.api:app --reload
```

In another terminal, start Vite:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Production-like local run

Build the React application, then let FastAPI serve it:

```bash
cd frontend
npm run build
cd ..
.venv/bin/uvicorn app.web.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

The API reads `GROQ_API_KEY` from the existing `.env` configuration. Token
counts shown in the run summary are tokenizer-independent estimates because
the structured-output provider currently does not expose usage metadata.
