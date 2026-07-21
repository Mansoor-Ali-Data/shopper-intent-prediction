import pandas as pd
from unittest.mock import Mock
import numpy as np

from shopper_intent.predict import ShopperIntentPredictor


def _sample_session():
    return pd.DataFrame(
        {
            "Administrative": [1],
            "Informational": [0],
            "ProductRelated": [5],
            "Administrative_Duration": [10.0],
            "Informational_Duration": [0.0],
            "ProductRelated_Duration": [120.0],
            "Month": ["Feb"],
            "VisitorType": ["Returning_Visitor"],
            "OperatingSystems": [1],
            "Browser": [2],
            "Region": [1],
            "TrafficType": [3],
            "Weekend": [False],
        }
    )


def test_predict_returns_expected_dictionary():
    predictor = ShopperIntentPredictor.__new__(ShopperIntentPredictor)

    predictor.model_name = "RandomForest"
    predictor.model_version = "1.0"
    predictor.preprocess = Mock(return_value=[[0.0]])

    predictor.model = Mock()
    predictor.model.predict_proba.return_value = [[0.2, 0.8]]

    result = predictor.predict(_sample_session())

    assert result["prediction"] == 1
    assert result["probability"] == 0.8
    assert result["model"] == "RandomForest"
    assert result["model_version"] == "1.0"


def test_predict_batch_returns_dataframe():
    predictor = ShopperIntentPredictor.__new__(ShopperIntentPredictor)

    predictor.model_name = "RandomForest"
    predictor.model_version = "1.0"
    predictor.preprocess = Mock(return_value=[[0.0], [0.0]])

    predictor.model = Mock()
    predictor.model.predict_proba.return_value = np.array([
        [0.3, 0.7],
        [0.9, 0.1],
    ]
    )
    
    result = predictor.predict_batch(_sample_session().iloc[[0, 0]])

    assert len(result) == 2
    assert list(result.columns) == [
        "prediction",
        "probability",
        "model",
        "model_version",
    ]
    assert result.loc[0, "prediction"] == 1
    assert result.loc[1, "prediction"] == 0


def test_prediction_probability_is_between_zero_and_one():
    predictor = ShopperIntentPredictor.__new__(ShopperIntentPredictor)

    predictor.model_name = "RandomForest"
    predictor.model_version = "1.0"
    predictor.preprocess = Mock(return_value=[[0.0]])

    predictor.model = Mock()
    predictor.model.predict_proba.return_value = [[0.4, 0.6]]

    result = predictor.predict(_sample_session())

    assert 0.0 <= result["probability"] <= 1.0