# Shopper Intent Prediction

Predict whether an online shopping session will result in revenue. The project
keeps the original exploratory notebook while providing a reproducible Python
training workflow and inference-ready model artifacts.

## Layout

- `src/shopper_intent/`: reusable data preparation, feature engineering,
  training, evaluation, and artifact persistence.
- `notebooks/`: the original exploration and model-development notebook.
- `configs/`: model and dataset configuration.
- `data/raw/`: source CSV; `data/processed/`: optional generated datasets.
- `models/`: generated model and preprocessing artifacts.
- `tests/`: automated checks for the preprocessing contract.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m shopper_intent.train --config configs/training.yaml
pytest
```

The training command reproduces the notebook's data preparation and model
parameters: duplicate removal, three engineered features, one-hot encoding,
stratified 80/20 split, `StandardScaler`, and training-only SMOTE. It writes
the fitted preprocessing bundle and each trained model to `models/`.

For inference, load `models/preprocessor.joblib` with
`shopper_intent.preprocessing.SessionPreprocessor`, transform raw session
records, then invoke the selected saved model. Use the notebook for EDA and
interactive visual analysis.
