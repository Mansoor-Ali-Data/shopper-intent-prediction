"""Training and inference preprocessing with a stable feature schema."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import add_engagement_features, encode_features


@dataclass
class SessionPreprocessor:
    """Replicate notebook preprocessing and retain its fitted feature order."""

    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_columns: list[str] = field(default_factory=list)

    def fit_transform(
        self, features: pd.DataFrame, feature_schema: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Fit scaling on `features` using an optional pre-split feature schema.

        The original notebook creates dummy columns before splitting the data.
        Supplying `feature_schema` retains that exact column layout while the
        scaler continues to be fitted only on the training partition.
        """
        encoded = encode_features(add_engagement_features(features))
        if feature_schema is not None:
            schema = encode_features(add_engagement_features(feature_schema))
            self.feature_columns = schema.columns.tolist()
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)
        else:
            self.feature_columns = encoded.columns.tolist()
        return self.scaler.fit_transform(encoded)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise RuntimeError("SessionPreprocessor must be fitted before transform.")
        encoded = encode_features(add_engagement_features(features))
        aligned = encoded.reindex(columns=self.feature_columns, fill_value=0)
        return self.scaler.transform(aligned)
