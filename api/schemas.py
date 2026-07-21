"""Pydantic schemas for the Shopper Intent Prediction API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for predicting online shopper purchase intent."""

    Administrative: int = Field(..., ge=0)
    Administrative_Duration: float = Field(..., ge=0.0)

    Informational: int = Field(..., ge=0)
    Informational_Duration: float = Field(..., ge=0.0)

    ProductRelated: int = Field(..., ge=0)
    ProductRelated_Duration: float = Field(..., ge=0.0)

    BounceRates: float = Field(..., ge=0.0)
    ExitRates: float = Field(..., ge=0.0)
    PageValues: float = Field(..., ge=0.0)
    SpecialDay: float = Field(..., ge=0.0)

    Month: str
    OperatingSystems: int = Field(..., ge=1)
    Browser: int = Field(..., ge=1)
    Region: int = Field(..., ge=1)
    TrafficType: int = Field(..., ge=1)

    VisitorType: str
    Weekend: bool


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    prediction: int
    probability: float
    model: str
    model_version: str