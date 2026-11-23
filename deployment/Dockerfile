# OmniSupport AI API. Build from repo root:  docker build -f deployment/Dockerfile -t omnisupport-ai .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src:/app
WORKDIR /app
COPY requirements-ci.txt requirements-serve.txt ./
RUN pip install -r requirements-serve.txt
COPY src ./src
COPY app ./app
COPY prompts ./prompts
COPY knowledge_base ./knowledge_base
COPY models ./models
COPY monitoring ./monitoring
COPY evaluation ./evaluation
COPY data/agent_store ./data/agent_store
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/logs && chown -R appuser /app/logs /app/monitoring
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
