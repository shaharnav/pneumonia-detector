from __future__ import annotations

import io
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from typing_extensions import TypedDict

from .model import PneumoniaClassifier, load_default_model


class PredictionResponse(TypedDict):
    label: Literal["Normal", "Pneumonia"]
    confidence: float


app = FastAPI(
    title="Pneumonia Detection API",
    description="Chest X-ray pneumonia detection using a fine-tuned ResNet-18.",
    version="0.1.0",
)

# Basic CORS settings – adjust allowed origins once you know your frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model() -> None:
    """
    Initialize and cache the model at application startup.
    """
    global classifier
    classifier = load_default_model()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accept a single chest X-ray image and return the predicted class
    and confidence score.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file (JPEG, PNG, etc.).",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not parse image file. Please upload a valid image.",
        )
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while reading image: {exc}",
        ) from exc

    label, confidence = classifier.predict_image(image)  # type: ignore[name-defined]

    return PredictionResponse(label=label, confidence=confidence)


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

