import pandas as pd

from shopper_intent.train import select_production_model


def test_selection_uses_weighted_score():
    evaluation = pd.DataFrame(
        {
            "Model": ["accurate_model", "fast_model"],
            "F1 Score": [0.90, 0.80],
            "Accuracy": [0.92, 0.89],
            "Precision": [0.91, 0.81],
            "Recall": [0.89, 0.79],
            "ROC AUC": [0.95, 0.85],
        }
    )

    benchmark = pd.DataFrame(
        {
            "Model": ["accurate_model", "fast_model"],
            "Average Latency (ms)": [10.0, 1.0],
            "Model Size (MB)": [5.0, 1.0],
        }
    )

    config = {
        "benchmark": {
            "selection": {
                "metric_weights": {
                    "f1_score": 0.50,
                    "roc_auc": 0.30,
                    "latency": 0.15,
                    "model_size": 0.05,
                }
            }
        }
    }

    selection = select_production_model(
        evaluation,
        benchmark,
        config,
    )

    assert selection["production_model"] == "accurate_model"
    assert "Overall Score" in selection["selection_reason"]


def test_selection_prioritizes_latency_when_configured():
    evaluation = pd.DataFrame(
        {
            "Model": ["accurate_model", "fast_model"],
            "F1 Score": [0.90, 0.80],
            "Accuracy": [0.92, 0.89],
            "Precision": [0.91, 0.81],
            "Recall": [0.89, 0.79],
            "ROC AUC": [0.95, 0.85],
        }
    )

    benchmark = pd.DataFrame(
        {
            "Model": ["accurate_model", "fast_model"],
            "Average Latency (ms)": [10.0, 1.0],
            "Model Size (MB)": [5.0, 1.0],
        }
    )

    config = {
        "benchmark": {
            "selection": {
                "metric_weights": {
                    "f1_score": 0.10,
                    "roc_auc": 0.10,
                    "latency": 0.70,
                    "model_size": 0.10,
                }
            }
        }
    }

    selection = select_production_model(
        evaluation,
        benchmark,
        config,
    )

    assert selection["production_model"] == "fast_model"