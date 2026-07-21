# 🛒 Shopper Intent Prediction System

An end-to-end Machine Learning Engineering project that predicts whether an online shopping session will result in a purchase. The project demonstrates the complete ML lifecycle, including data preprocessing, feature engineering, model training, benchmarking, production model selection, REST API deployment, monitoring, and automated testing.

---

## 🚀 Project Highlights

- End-to-end machine learning pipeline
- Automated preprocessing and feature engineering
- Multiple classification models
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Artificial Neural Network (ANN)
- Weighted production model selection
- FastAPI prediction service
- Data quality and feature drift monitoring
- Comprehensive unit testing using Pytest
- Modular and configuration-driven architecture

---

## 🏗️ System Architecture

<p align="center">
    <img src="docs/images/system_architecture.png" width="900">
</p>

---

## 📂 Repository Structure

```text
.
├── api/                    # FastAPI application
├── configs/                # Training configuration
├── data/
│   ├── raw/
│   └── processed/
├── docs/                   # Project documentation
├── models/                 # Trained models and deployment artifacts
├── reports/                # Monitoring reports
├── src/
│   └── shopper_intent/     # ML pipeline implementation
├── tests/                  # Unit tests
├── README.md
└── pyproject.toml
```

---

## ⚙️ Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd shopper-intent-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
poetry install
```

### 3. Train the models

```bash
python -m src.shopper_intent.train
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🌐 API Example

### Request

```http
POST /predict
```

```json
{
    "Administrative": 2,
    "Administrative_Duration": 45.0,
    "Informational": 1,
    "Informational_Duration": 12.5,
    "ProductRelated": 18,
    "ProductRelated_Duration": 620.3,
    "BounceRates": 0.02,
    "ExitRates": 0.05,
    "PageValues": 18.7,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 1,
    "Region": 3,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
}
```

### Response

```json
{
    "prediction": 1,
    "probability": 0.9437,
    "model": "logistic_regression",
    "model_version": "1.0.0"
}
```

---

## 🧪 Running Tests

Execute the unit test suite using:

```bash
pytest
```

Verbose output:

```bash
pytest -v
```

---

# 📚 Documentation

Detailed technical documentation is available in the **`docs/`** directory.

| Document | Description |
|----------|-------------|
| 📖 **[01. System Design](docs/01_system_design.md)** | System architecture, component design, project structure, data flow, and design decisions. |
| 🤖 **[02. Model Development](docs/02_model_development.md)** | Dataset, preprocessing, feature engineering, model development, evaluation, benchmarking, production model selection, and model artifacts. |
| 📊 **[03. Monitoring Strategy](docs/03_monitoring_strategy.md)** | Data quality monitoring, feature drift detection, retraining recommendation, monitoring reports, and limitations. |
| 🌐 **[04. API Documentation](docs/04_api_documentation.md)** | REST API endpoints, request and response schemas, validation, and deployment instructions. |
| 🧪 **[05. Testing](docs/05_testing.md)** | Testing strategy, unit test coverage, execution instructions, and testing limitations. |

---

## 📖 Documentation Roadmap

```text
README
│
├── 01. System Design
│
├── 02. Model Development
│
├── 03. Monitoring Strategy
│
├── 04. API Documentation
│
└── 05. Testing
```

The README provides a high-level overview of the project, while the `docs/` directory contains detailed technical documentation for each stage of the machine learning engineering workflow.

---

## 📈 Technologies

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, TensorFlow/Keras |
| Data Processing | Pandas, NumPy |
| API | FastAPI |
| Testing | Pytest |
| Configuration | YAML |
| Model Serialization | Joblib |

---

## 📄 License

This project was developed for educational purposes as part of a Machine Learning Engineering assignment.