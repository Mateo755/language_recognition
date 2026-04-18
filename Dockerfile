# syntax=docker/dockerfile:1
# Inference-only image: FastAPI + shared text cleaning + model artifact.
# Copy models/trained_pipeline-0.1.0.pkl to app/model/ before build, or mount and set
# LANGUAGE_DETECTION_MODEL_PATH.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /code

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Minimal shared code: clean_text must match training (see src/text_cleaning.py)
COPY src/__init__.py src/text_cleaning.py ./src/

COPY app ./app

ENV PYTHONPATH=/code

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
