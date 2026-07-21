"""FastAPI application for Shopper Intent Prediction."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI

from api.schemas import PredictionRequest, PredictionResponse
from src.shopper_intent.predict import ShopperIntentPredictor

app = FastAPI(
    title="Shopper Intent Prediction API",
    description="Predict whether an online shopping session will result in a purchase.",
    version="1.0.0",
)

# Load the model once when the application starts
predictor = ShopperIntentPredictor()


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "service": "Shopper Intent Prediction API",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict whether a shopping session will end in a purchase.
    """

    data = pd.DataFrame([request.model_dump()])

    result = predictor.predict(data)

    return PredictionResponse(**result)