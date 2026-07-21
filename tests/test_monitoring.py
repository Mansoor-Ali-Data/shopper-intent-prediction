import pandas as pd
import pytest

from shopper_intent.monitoring import (
    check_data_quality,
    detect_feature_drift,
    should_retrain,
)


def test_check_data_quality_passes_clean_data():
    data = pd.DataFrame(
        {
            "BounceRates": [0.01, 0.02],
            "ExitRates": [0.03, 0.04],
            "PageValues": [5.0, 10.0],
        }
    )

    result = check_data_quality(data)

    assert result["status"] == "PASS"
    assert result["warnings"] == []


def test_check_data_quality_detects_missing_values():
    data = pd.DataFrame(
        {
            "BounceRates": [0.01, None],
            "ExitRates": [0.03, 0.04],
            "PageValues": [5.0, 10.0],
        }
    )

    result = check_data_quality(data)

    assert result["status"] == "WARNING"
    assert any("Missing values" in warning for warning in result["warnings"])


def test_detect_feature_drift_identifies_shift():
    training = pd.DataFrame(
        {
            "BounceRates": [0.02] * 100,
            "ExitRates": [0.04] * 100,
        }
    )

    recent = pd.DataFrame(
        {
            "BounceRates": [0.05] * 100,
            "ExitRates": [0.04] * 100,
        }
    )

    result = detect_feature_drift(training, recent)
    
    
    assert result["drift_detected"] is True
    assert "BounceRates" in result["drifted_features"]

    stats = result["statistics"]["BounceRates"]

    assert stats["train_mean"] == pytest.approx(0.02)
    assert stats["recent_mean"] == pytest.approx(0.05)
    assert stats["mean_shift"] > 0


def test_should_retrain_when_drift_detected():
    assert should_retrain(
        drift_detected=True,
        accuracy_drop=0.0,
        new_data_days=1,
    )


def test_should_retrain_when_accuracy_drops():
    assert should_retrain(
        drift_detected=False,
        accuracy_drop=0.10,
        new_data_days=1,
    )


def test_should_retrain_after_retrain_interval():
    assert should_retrain(
        drift_detected=False,
        accuracy_drop=0.0,
        new_data_days=35,
    )


def test_should_not_retrain_when_no_conditions_met():
    assert not should_retrain(
        drift_detected=False,
        accuracy_drop=0.01,
        new_data_days=7,
    )