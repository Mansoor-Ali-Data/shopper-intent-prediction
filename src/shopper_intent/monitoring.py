"""Monitoring utilities for data quality, drift detection and retraining."""

from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================

from typing import Any

import pandas as pd

from .features import NUMERIC_FEATURES

import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

NEGATIVE_VALUE_COLUMNS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# ============================================================================
# Data Quality Monitoring
# ============================================================================

def check_data_quality(data: pd.DataFrame) -> dict[str, Any]:
    """
    Perform lightweight data quality checks.

    Checks
    ------
    - Missing values
    - Duplicate rows
    - Negative numeric values

    Returns
    -------
    dict
        Data quality report.
    """

    warnings: list[str] = []

    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        warnings.append(
            f"Missing values detected in columns: {missing.to_dict()}"
        )

    duplicates = int(data.duplicated().sum())
    if duplicates > 0:
        warnings.append(f"{duplicates} duplicate rows detected.")

    for column in NEGATIVE_VALUE_COLUMNS:
        if column in data.columns:
            count = int((data[column] < 0).sum())
            if count > 0:
                warnings.append(
                    f"{count} negative values found in '{column}'."
                )

    status = "PASS" if not warnings else "WARNING"

    return {
        "status": status,
        "warnings": warnings,
    }

# ============================================================================
# Feature Drift Detection
# ============================================================================

def detect_feature_drift(
    training_data: pd.DataFrame,
    recent_data: pd.DataFrame,
    threshold: float = 0.20,
) -> dict[str, Any]:
    """
    Compare training and recent data using simple mean/std statistics.
    """

    drifted_features: list[str] = []
    statistics: dict[str, dict[str, float]] = {}

    for feature in NUMERIC_FEATURES:

        if (
            feature not in training_data.columns
            or feature not in recent_data.columns
        ):
            continue

        train_mean = training_data[feature].mean()
        recent_mean = recent_data[feature].mean()

        train_std = training_data[feature].std()
        recent_std = recent_data[feature].std()

        mean_shift = abs(recent_mean - train_mean) / (
            abs(train_mean) + 1e-6
        )

        std_shift = abs(recent_std - train_std) / (
            abs(train_std) + 1e-6
        )

        if mean_shift > threshold or std_shift > threshold:

            drifted_features.append(feature)

            statistics[feature] = {
                "train_mean": float(train_mean),
                "recent_mean": float(recent_mean),
                "train_std": float(train_std),
                "recent_std": float(recent_std),
                "mean_shift": float(mean_shift),
                "std_shift": float(std_shift),
            }

    status = "PASS" if not drifted_features else "WARNING"

    return {
        "status": status,
        "drift_detected": bool(drifted_features),
        "drifted_features": drifted_features,
        "statistics": statistics,
    }

# ============================================================================
# Retraining Decision
# ============================================================================

def should_retrain(
    *,
    drift_detected: bool,
    accuracy_drop: float,
    new_data_days: int,
    accuracy_threshold: float = 0.05,
    retrain_days: int = 30,
) -> bool:
    """
    Decide whether the model should be retrained.

    Retraining is recommended when:
    - Feature drift is detected.
    - Accuracy drops beyond the threshold.
    - Enough new data has accumulated.
    """

    if drift_detected:
        return True

    if accuracy_drop >= accuracy_threshold:
        return True

    if new_data_days >= retrain_days:
        return True

    return False

# ============================================================================
# Monitoring Report Generation
# ============================================================================

def generate_monitoring_report(
    training_data: pd.DataFrame,
    recent_data: pd.DataFrame,
    accuracy_drop: float = 0.0,
    new_data_days: int = 0,
) -> dict[str, Any]:
    """
    Run all monitoring checks and generate a unified report.
    """

    quality = check_data_quality(recent_data)

    drift = detect_feature_drift(
        training_data=training_data,
        recent_data=recent_data,
    )

    retrain = should_retrain(
        drift_detected=drift["drift_detected"],
        accuracy_drop=accuracy_drop,
        new_data_days=new_data_days,
    )

    if drift["drift_detected"]:
        reason = "Feature drift detected."
    elif accuracy_drop >= 0.05:
        reason = "Model accuracy dropped."
    elif new_data_days >= 30:
        reason = "New training data available."
    else:
        reason = "Monitoring checks passed."

    return {
        "data_quality": quality,
        "feature_drift": drift,
        "retraining": {
            "recommended": retrain,
            "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "batch_size": len(recent_data),
        },
    }

# ============================================================================
# Report Display
# ============================================================================

def print_monitoring_report(
    report: dict[str, Any],
    report_path: str | Path = "reports/monitoring_report.json",
) -> None:
    """
    Print a formatted monitoring report.
    """

    print("\n" + "=" * 50)
    print("MODEL MONITORING REPORT")
    print("=" * 50)

    # ------------------------------------------------------------------------
    # Data Quality
    # ------------------------------------------------------------------------

    quality = report["data_quality"]

    print("\nDATA QUALITY")
    print("-" * 50)
    print(f"Status : {quality['status']}")

    if quality["warnings"]:
        print("\nWarnings:")
        for warning in quality["warnings"]:
            print(f" - {warning}")

    # ------------------------------------------------------------------------
    # Feature Drift
    # ------------------------------------------------------------------------

    drift = report["feature_drift"]

    print("\nFEATURE DRIFT")
    print("-" * 50)
    print(f"Status : {drift['status']}")

    if drift["drifted_features"]:

        print("\nFeature Statistics")

        for feature, stats in drift["statistics"].items():
            print(f"\n{feature}")
            print(f"  Train Mean : {stats['train_mean']:.4f}")
            print(f"  Recent Mean: {stats['recent_mean']:.4f}")
            print(f"  Mean Shift : {stats['mean_shift']:.2%}")
            print(f"  Std Shift  : {stats['std_shift']:.2%}")

    else:
        print("No feature drift detected.")

    # ------------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------------

    retraining = report["retraining"]

    print("\nRETRAINING")
    print("-" * 50)
    print(
        f"Recommended : {'YES' if retraining['recommended'] else 'NO'}"
    )
    print(f"Reason      : {retraining['reason']}")
    print(f"Batch Size  : {retraining['batch_size']}")
    print(f"Timestamp   : {retraining['timestamp']}")

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(f"✓ Data Quality : {quality['status']}")

    if drift["drift_detected"]:
        print(
            f"⚠ Feature Drift : DETECTED "
            f"({len(drift['drifted_features'])} features)"
        )
    else:
        print("✓ Feature Drift : NOT DETECTED")

    print(f"✓ Report Saved  : {report_path}")

# ============================================================================
# Report Persistence
# ============================================================================

def save_monitoring_report(
    report: dict[str, Any],
    output_path: str | Path = "reports/monitoring_report.json",
) -> None:
    """
    Save the monitoring report as JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        
        json.dump(report, f, indent=4)


# ============================================================================
# Demonstration
# ============================================================================

def main() -> None:
    """
    Demonstrate monitoring on a simulated production batch.
    """

    data = pd.read_csv("data/raw/online_shoppers_intention.csv")

    training_data = data.copy()

    recent_data = data.sample(500, random_state=42).copy()

    # Simulate production drift
    recent_data["BounceRates"] *= 1.35
    recent_data["ExitRates"] *= 1.20

    report = generate_monitoring_report(
        training_data=training_data,
        recent_data=recent_data,
        accuracy_drop=0.02,
        new_data_days=15,
    )

    report_path = "reports/monitoring_report.json"

    print_monitoring_report(report, report_path)

    save_monitoring_report(report, report_path)

# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    
    main()
