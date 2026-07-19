import pandas as pd
import pytest

from shopper_intent.preprocessing import SessionPreprocessor


def _sessions():
    return pd.DataFrame({
        "Administrative": [0, 1], "Informational": [0, 1], "ProductRelated": [1, 3],
        "Administrative_Duration": [0.0, 5.0], "Informational_Duration": [0.0, 5.0],
        "ProductRelated_Duration": [10.0, 30.0], "Month": ["Feb", "Mar"],
        "VisitorType": ["Returning_Visitor", "New_Visitor"], "OperatingSystems": [1, 2],
        "Browser": [1, 2], "Region": [1, 2], "TrafficType": [1, 2],
        "Weekend": [False, True],
    })


def test_transform_keeps_training_schema_for_unseen_categories():
    preprocessor = SessionPreprocessor()
    training = _sessions()
    transformed_train = preprocessor.fit_transform(training)
    incoming = training.iloc[[0]].copy()
    incoming.loc[:, "Month"] = "June"

    transformed_incoming = preprocessor.transform(incoming)

    assert transformed_train.shape[1] == transformed_incoming.shape[1]


def test_presplit_schema_preserves_notebook_dummy_columns():
    preprocessor = SessionPreprocessor()
    all_sessions = _sessions()
    training_only = all_sessions.iloc[[0]]

    transformed = preprocessor.fit_transform(
        training_only, feature_schema=all_sessions
    )

    assert "Month_Mar" in preprocessor.feature_columns
    assert transformed.shape[1] == len(preprocessor.feature_columns)


def test_transform_requires_fitted_preprocessor():
    with pytest.raises(RuntimeError):
        SessionPreprocessor().transform(_sessions())
