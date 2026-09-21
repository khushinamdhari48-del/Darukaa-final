FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    BIOAI_INDEX_DIR=/app/.index

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Build the retrieval index at image-build time so the container starts warm and
# a malformed evidence card fails the build rather than the first request.
RUN python scripts/build_index.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

# Honour $PORT so the same image runs on Render, Railway, Fly and Cloud Run.
CMD ["sh", "-c", "uvicorn bioai.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
