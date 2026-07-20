"""Production-style orchestration for shopper-intent model training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.parsers.expat import model

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .benchmark import benchmark_models as benchmark_model_artifacts
from .benchmark import predict_probabilities
from .evaluation import binary_metrics
from .preprocessing import SessionPreprocessor


@dataclass
class PreparedData:
    """Data partitions and fitted preprocessing state used by training."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: SessionPreprocessor


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the pipeline configuration from YAML."""
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load raw data and retain the notebook's duplicate-removal behavior."""
    return pd.read_csv(config["data"]["raw_path"]).drop_duplicates()


def prepare_data(frame: pd.DataFrame, config: dict[str, Any]) -> PreparedData:
    """Split, transform, scale, and balance data using existing project logic."""
    data_config = config["data"]
    target = data_config["target_column"]
    features = frame.drop(columns=target)
    labels = frame[target].astype(int)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=labels,
    )
    preprocessor = SessionPreprocessor()
    x_train_scaled = preprocessor.fit_transform(
        x_train_raw, feature_schema=features
    )
    x_test_scaled = preprocessor.transform(x_test_raw)
    x_train_smote, y_train_smote = SMOTE(
        random_state=data_config["random_state"]
    ).fit_resample(x_train_scaled, y_train)
    return PreparedData(
        x_train=x_train_smote,
        x_test=x_test_scaled,
        y_train=y_train_smote,
        y_test=y_test,
        preprocessor=preprocessor,
    )


def _keras_model(input_size: int, patience: int) -> tuple[dict[str, Any], Any]:
    """Create the production ANN model with Early Stopping."""
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore
    from tensorflow.keras.layers import Dense # type: ignore
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore

    ann = Sequential(
        [
            Dense(32, activation="relu", input_shape=(input_size,)),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    ann.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    return {"ann": ann}, callback


def create_models(input_size: int, config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Create every supported model and return them in one named dictionary."""
    training_config = config["training"]
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }
    ann_model, callback = _keras_model(
        input_size, training_config["early_stopping_patience"]
    )
    models.update(ann_model)
    return models, callback


def train_models(prepared: PreparedData, config: dict[str, Any]) -> dict[str, Any]:
    """Fit all existing classifiers while preserving their training settings."""
    training_config = config["training"]
    models, early_stopping_callback = create_models(prepared.x_train.shape[1], config)
    for name, model in models.items():
        if name == "ann":
            model.fit(
                prepared.x_train,
                prepared.y_train,
                validation_split=0.20,
                epochs=training_config["ann_epochs"],
                batch_size=training_config["ann_batch_size"],
                callbacks=[early_stopping_callback],
                verbose=1,
            )
        else:
            model.fit(prepared.x_train, prepared.y_train)
    return models


def evaluate_models(models: dict[str, Any], prepared: PreparedData) -> pd.DataFrame:
    """Evaluate all fitted models with the existing classification metrics."""
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        probabilities = predict_probabilities(model, prepared.x_test)
        predictions = (probabilities > 0.5).astype(int) if name.endswith("ann") else model.predict(prepared.x_test)
        rows.append({"Model": name, **binary_metrics(prepared.y_test, predictions, probabilities)})
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False)


def save_model_artifacts(
    models: dict[str, Any], preprocessor: SessionPreprocessor, output_dir: Path
) -> dict[str, Path]:
    """Persist models and the matching preprocessing artifact for later inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    for name, model in models.items():
        suffix = ".keras" if name.endswith("ann") else ".joblib"
        path = output_dir / f"{name}{suffix}"
        if suffix == ".keras":
            model.save(path)
        else:
            joblib.dump(model, path)
        artifact_paths[name] = path
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    return artifact_paths


def benchmark_models(
    models: dict[str, Any], artifact_paths: dict[str, Path], prepared: PreparedData, config: dict[str, Any]
) -> pd.DataFrame:
    """Benchmark model-only single-request latency and serialized size."""
    benchmark_config = config["benchmark"]
    return benchmark_model_artifacts(
        models=models,
        artifact_paths=artifact_paths,
        representative_features=prepared.x_test,
        repetitions=benchmark_config["latency_repetitions"],
        warmup_runs=benchmark_config["warmup_runs"],
    )


def select_production_model(
    evaluation: pd.DataFrame, benchmark: pd.DataFrame, config: dict[str, Any]
) -> dict[str, str]:
    """Select the production model using a weighted score based on
    predictive performance and deployment characteristics."""
    
    print(evaluation.columns.tolist())
    
    weights = config["benchmark"]["selection"]["metric_weights"]

    combined = evaluation.merge(
        benchmark,
        on="Model",
        validate="one_to_one",
    ).copy()

    # -----------------------------
    # Normalize metrics to [0, 1]
    # -----------------------------

    # Higher is better
    combined["F1 Score (Norm)"] = (
        (combined["F1 Score"] - combined["F1 Score"].min())
        / (combined["F1 Score"].max() - combined["F1 Score"].min() + 1e-9)
    )

    combined["ROC AUC (Norm)"] = (
        (combined["ROC AUC"] - combined["ROC AUC"].min())
        / (
            combined["ROC AUC"].max()
            - combined["ROC AUC"].min()
            + 1e-9
        )
    )

    # Lower is better
    combined["Latency (Norm)"] = 1 - (
        (combined["Average Latency (ms)"] - combined["Average Latency (ms)"].min())
        / (
            combined["Average Latency (ms)"].max()
            - combined["Average Latency (ms)"].min()
            + 1e-9
        )
    )

    combined["Model Size (Norm)"] = 1 - (
        (combined["Model Size (MB)"] - combined["Model Size (MB)"].min())
        / (
            combined["Model Size (MB)"].max()
            - combined["Model Size (MB)"].min()
            + 1e-9
        )
    )

    # -----------------------------
    # Weighted score
    # -----------------------------
    combined["Overall Score"] = (
        weights["f1_score"] * combined["F1 Score (Norm)"]
        + weights["roc_auc"] * combined["ROC AUC (Norm)"]
        + weights["latency"] * combined["Latency (Norm)"]
        + weights["model_size"] * combined["Model Size (Norm)"]
    )

    selected = combined.sort_values(
        "Overall Score", ascending=False
    ).iloc[0]

    reason = (
        f"Selected using weighted model selection "
        f"(F1={weights['f1_score']:.0%}, "
        f"Latency={weights['latency']:.0%}, "
        f"ROC-AUC={weights['roc_auc']:.0%}, "
        f"Model Size={weights['model_size']:.0%}). "
        f"Overall Score = {selected['Overall Score']:.4f}."
    )

    return {
        "production_model": str(selected["Model"]),
        "selection_reason": reason,
    }

def save_artifacts(
    evaluation: pd.DataFrame,
    benchmark: pd.DataFrame,
    selection: dict[str, str],
    artifact_paths: dict[str, Path],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Save metrics and deployment metadata after model selection."""
    evaluation.to_csv(output_dir / "model_metrics.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    deployment = {
        "production_model": selection["production_model"],
        "model_path": artifact_paths[selection["production_model"]].as_posix(),
        "preprocessor_path": (output_dir / "preprocessor.joblib").as_posix(),
        "model_version": config["deployment"]["model_version"],
        "selection_reason": selection["selection_reason"],
    }
    with (output_dir / "deployment.json").open("w", encoding="utf-8") as deployment_file:
        json.dump(deployment, deployment_file, indent=2)
    return deployment


def generate_training_report(
    evaluation: pd.DataFrame,
    benchmark: pd.DataFrame,
    deployment: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write a Markdown summary of model quality, engineering metrics, and choice."""
    combined = evaluation.merge(benchmark, on="Model", validate="one_to_one")
    report = [
        "# Training Summary",
        "",
        "## Evaluation and Engineering Metrics",
        "",
        combined.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Production Model",
        "",
        f"Selected model: `{deployment['production_model']}`",
        "",
        deployment["selection_reason"],
        "",
    ]
    report_path = output_dir / "training_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def run_training(config: dict[str, Any]) -> pd.DataFrame:
    """Run the full training, evaluation, benchmark, selection, and save workflow."""
    frame = load_dataset(config)
    prepared = prepare_data(frame, config)
    models = train_models(prepared, config)
    evaluation = evaluate_models(models, prepared)
    output_dir = Path(config["training"]["output_dir"])
    artifact_paths = save_model_artifacts(models, prepared.preprocessor, output_dir)
    benchmark = benchmark_models(models, artifact_paths, prepared, config)
    selection = select_production_model(evaluation, benchmark, config)
    deployment = save_artifacts(
        evaluation, benchmark, selection, artifact_paths, output_dir, config
    )
    generate_training_report(evaluation, benchmark, deployment, output_dir)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    evaluation = run_training(load_config(args.config))
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
