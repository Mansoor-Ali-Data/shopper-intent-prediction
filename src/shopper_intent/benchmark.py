"""Engineering benchmarks for trained model artifacts."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def predict_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities through one interface.

    Scikit-learn classifiers expose ``predict_proba`` while Keras models expose
    ``predict``. Keeping this difference here prevents framework-specific
    branches throughout evaluation and benchmark orchestration.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    return np.asarray(model.predict(features, verbose=0)).reshape(-1)


def benchmark_models(
    models: dict[str, Any],
    artifact_paths: dict[str, Path],
    representative_features: np.ndarray,
    repetitions: int,
    warmup_runs: int = 10,
) -> pd.DataFrame:
    """Measure average prediction latency and saved artifact size per model."""
    if repetitions < 1:
        raise ValueError("Benchmark repetitions must be at least one.")
    if len(representative_features) == 0:
        raise ValueError("Benchmark features must contain at least one sample.")

    sample = representative_features[:1]
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        for _ in range(warmup_runs):
            predict_probabilities(model, sample)

        durations_ms = []
        for _ in range(repetitions):
            start = perf_counter()
            predict_probabilities(model, sample)
            durations_ms.append((perf_counter() - start) * 1_000)

        artifact_path = artifact_paths[name]
        rows.append(
            {
                "Model": name,
                "Average Latency (ms)": float(np.mean(durations_ms)),
                "Model Size (MB)": artifact_path.stat().st_size / (1024**2),
            }
        )
    return pd.DataFrame(rows).sort_values("Average Latency (ms)")
