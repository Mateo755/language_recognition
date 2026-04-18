"""FastAPI service for language detection using the trained sklearn pipeline."""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_preprocessing import clean_text

logger = logging.getLogger(__name__)

# Packaged with the app (e.g. COPY app/model/*.pkl in Docker); sibling of main.py
_APP_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _APP_DIR / "model" / "trained_pipeline-0.1.0.pkl"
_MODEL_ENV_VAR = "LANGUAGE_DETECTION_MODEL_PATH"
_MAX_TEXT_CHARS = 50_000

model_pipeline: Optional[Any] = None


def _resolve_model_path() -> Path:
    """Return absolute path to the pickle file; env overrides bundled default under app/model/."""
    override = os.environ.get(_MODEL_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_MODEL_PATH.resolve()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the pickled pipeline once at startup; fail fast if the file is missing."""
    global model_pipeline
    path = _resolve_model_path()
    if not path.is_file():
        msg = (
            f"Model file not found: {path}. "
            f"Train the model or set {_MODEL_ENV_VAR} to a valid .pkl path."
        )
        logger.error(msg)
        raise RuntimeError(msg)
    with path.open("rb") as f:
        # Trusted local artifact from this repo's training script; do not load untrusted pickles.
        model_pipeline = pickle.load(f)
    logger.info("Loaded language detection model from %s", path)
    yield
    model_pipeline = None


app = FastAPI(
    title="Language Detector API",
    version="1.0",
    lifespan=lifespan,
)


class TextRequest(BaseModel):
    """Request body for POST /detect."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_TEXT_CHARS,
        description="Raw text to classify (same preprocessing as training).",
    )


class DetectionResponse(BaseModel):
    """JSON response for a single prediction."""

    text: str
    language: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: process is up and model was loaded at startup."""
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
def detect_language(request: TextRequest) -> DetectionResponse:
    """Run the trained pipeline on cleaned text and return the predicted language label."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )
    cleaned = clean_text(request.text)
    prediction = model_pipeline.predict([cleaned])
    language = str(prediction[0])
    return DetectionResponse(text=request.text, language=language)
