from pathlib import Path

import numpy as np

from shopper_intent.benchmark import benchmark_models, predict_probabilities


class SklearnLikeModel:
    def predict_proba(self, features):
        return np.column_stack([np.zeros(len(features)), np.ones(len(features))])


class KerasLikeModel:
    def predict(self, features, verbose=0):
        return np.ones((len(features), 1))


def test_prediction_adapter_supports_sklearn_and_keras_interfaces():
    sample = np.array([[1.0, 2.0]])

    assert predict_probabilities(SklearnLikeModel(), sample).tolist() == [1.0]
    assert predict_probabilities(KerasLikeModel(), sample).tolist() == [1.0]


def test_benchmark_returns_latency_and_artifact_size():
    artifact = Path(__file__)

    result = benchmark_models(
        models={"test_model": SklearnLikeModel()},
        artifact_paths={"test_model": artifact},
        representative_features=np.array([[1.0, 2.0]]),
        repetitions=2,
        warmup_runs=0,
    )

    assert result.loc[0, "Model"] == "test_model"
    assert result.loc[0, "Average Latency (ms)"] >= 0
    assert result.loc[0, "Model Size (MB)"] > 0
