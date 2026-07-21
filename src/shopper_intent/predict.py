"""Inference utilities for the deployed shopper intent model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from tensorflow.keras.models import load_model                                                                                                    #type: ignore

from .features import add_engagement_features
from .preprocessing import SessionPreprocessor


class ShopperIntentPredictor:
    """Load the deployed model and perform inference."""

    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)

        deployment_path = self.model_dir / "deployment.json"

        with deployment_path.open("r", encoding="utf-8") as file:
            deployment = json.load(file)

        self.model_name = deployment["production_model"]
        self.model_version = deployment.get("model_version", "unknown")

        model_path = Path(deployment["model_path"])
        preprocessor_path = Path(deployment["preprocessor_path"])

        if model_path.suffix == ".keras":
            self.model = load_model(model_path)
        else:
            self.model = joblib.load(model_path)

        self.preprocessor: SessionPreprocessor = joblib.load(preprocessor_path)

    def preprocess(self, data: pd.DataFrame):
        """
        Apply feature engineering and preprocessing.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data.

        Returns
        -------
        np.ndarray
            Preprocessed feature matrix.
        """
        engineered = add_engagement_features(data)
        return self.preprocessor.transform(engineered)

    def predict(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Predict purchase intent.

        Parameters
        ----------
        data : pd.DataFrame
            Raw customer session data.

        Returns
        -------
        dict
            Prediction results.
        """
        X = self.preprocess(data)

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X)[0][1])
        else:
            probability = float(self.model.predict(X, verbose=0)[0][0])

        prediction = int(probability >= 0.5)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model": self.model_name,
            "model_version": self.model_version,
        }

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict purchase intent for multiple customer sessions.

        Parameters
        ----------
        data : pd.DataFrame
            Raw customer session data.

        Returns
        -------
        pd.DataFrame
            A dataframe containing predictions and probabilities.
        """
        X = self.preprocess(data)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = self.model.predict(X, verbose=0).flatten()

        predictions = (probabilities >= 0.5).astype(int)

        return pd.DataFrame(
            {
                "prediction": predictions,
                "probability": probabilities.round(4),
                "model": self.model_name,
                "model_version": self.model_version,
            }
        )

if __name__ == "__main__":
    print("ShopperIntentPredictor module.")
    print("Import this class or use it through the FastAPI service.")