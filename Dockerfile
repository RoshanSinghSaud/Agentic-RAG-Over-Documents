FROM python:3.11-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Everything else, filtered by .dockerignore (app.py, src/, requirements.txt
# survive; the venv, eval/, notebooks/, .git, secrets, and runtime state don't).
COPY . .

EXPOSE 8000

# python, not curl — keeps the image from needing an extra apt-get layer.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
