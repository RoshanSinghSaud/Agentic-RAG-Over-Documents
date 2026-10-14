# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements.lock.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install --timeout 120 --retries 10 -r requirements.lock.txt

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .
# Bakes the papers into the image so a Render cold start never depends on
# arXiv responding. Runs before any secrets exist, hence a script that only
# imports src.corpus (stdlib-only) rather than src.ingestion, which pulls in
# src.config and exits without OPENAI_API_KEY.
RUN python scripts/fetch_corpus.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/ready', timeout=3)" || exit 1
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]