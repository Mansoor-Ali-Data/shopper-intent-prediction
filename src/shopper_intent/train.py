"""Reproducible training entry point matching the original notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .evaluation import binary_metrics
from .preprocessing import SessionPreprocessor


def _keras_models(input_size: int, patience: int):
    """Create the original baseline and two ANN experiment architectures."""
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    baseline = Sequential([Dense(32, activation="relu", input_shape=(input_size,)), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])
    larger = Sequential([Dense(64, activation="relu", input_shape=(input_size,)), Dense(32, activation="relu"), Dense(1, activation="sigmoid")])
    early_stopping = Sequential([Dense(32, activation="relu", input_shape=(input_size,)), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])
    baseline.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    for model in (larger, early_stopping):
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    callback = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return baseline, larger, early_stopping, callback


def run_training(config: dict) -> pd.DataFrame:
    """Train, evaluate, and persist all models from the notebook."""
    data_config = config["data"]
    training_config = config["training"]
    frame = pd.read_csv(data_config["raw_path"]).drop_duplicates()
    target = data_config["target_column"]
    features, labels = frame.drop(columns=target), frame[target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=data_config["test_size"],
        random_state=data_config["random_state"], stratify=labels,
    )
    preprocessor = SessionPreprocessor()
    x_train_scaled = preprocessor.fit_transform(x_train, feature_schema=features)
    x_test_scaled = preprocessor.transform(x_test)
    x_train_smote, y_train_smote = SMOTE(random_state=data_config["random_state"]).fit_resample(x_train_scaled, y_train)

    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    results: list[dict[str, float | str]] = []
    for name, model in models.items():
        model.fit(x_train_smote, y_train_smote)
        probabilities = model.predict_proba(x_test_scaled)[:, 1]
        result = {"Model": name, **binary_metrics(y_test, model.predict(x_test_scaled), probabilities)}
        results.append(result)
        joblib.dump(model, output_dir / f"{name}.joblib")

    baseline, larger, early_stop_model, callback = _keras_models(
        x_train_smote.shape[1], training_config["early_stopping_patience"]
    )
    ann_runs = [
        ("baseline_ann", baseline, training_config["ann_epochs"], None),
        ("large_ann", larger, training_config["ann_epochs"], None),
        ("early_stopping_ann", early_stop_model, training_config["early_stopping_epochs"], [callback]),
    ]
    for name, model, epochs, callbacks in ann_runs:
        model.fit(x_train_smote, y_train_smote, validation_split=0.20, epochs=epochs, batch_size=training_config["ann_batch_size"], callbacks=callbacks, verbose=1)
        probabilities = model.predict(x_test_scaled, verbose=0).ravel()
        predictions = (probabilities > 0.5).astype(int)
        results.append({"Model": name, **binary_metrics(y_test, predictions, probabilities)})
        model.save(output_dir / f"{name}.keras")

    comparison = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    comparison.to_csv(output_dir / "model_metrics.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        comparison = run_training(yaml.safe_load(config_file))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
