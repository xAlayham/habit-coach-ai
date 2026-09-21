FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN python -m venv /opt/venv

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY rag.py ingest.py ./
COPY knowledge ./knowledge
RUN python ingest.py --rebuild


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    PATH="/opt/venv/bin:$PATH" \
    CHROMA_DIR=/app/chroma_db \
    PORT=8000

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/chroma /home/appuser/.cache/chroma
COPY --from=builder /app/chroma_db /app/chroma_db

COPY agent.py service.py rag.py ratelimit.py lru.py ingest.py ./
COPY knowledge ./knowledge

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "exec uvicorn service:app --host 0.0.0.0 --port ${PORT}"]
