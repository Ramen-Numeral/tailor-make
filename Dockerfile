FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    AI_DETECTION_ENABLED=false \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user

WORKDIR /home/user/app

EXPOSE 7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libcairo2 \
        libffi-dev \
        libgdk-pixbuf-2.0-0 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user app ./app
COPY --chown=user:user config ./config
COPY --chown=user:user ml_pipelines ./ml_pipelines
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./frontend/dist

USER user

# Download trained detector weights into the application model directory.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ramen-numeral/tailormake-detector', local_dir='models')"

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["sh", "-c", "exec uvicorn app.web.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
