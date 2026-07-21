# 1. Introduction

The Shopper Intent Prediction API provides a RESTful interface for serving the trained machine learning model. Built using FastAPI, the service accepts customer browsing session information, applies the same preprocessing and feature engineering pipeline used during model training, and returns a prediction indicating whether the session is likely to result in a purchase.

When the application starts, the production model and preprocessing pipeline are loaded once into memory. This reduces prediction latency by avoiding repeated model loading for each request and ensures that all predictions are generated using the production model specified in the deployment configuration.

The API is intended for local deployment and demonstration purposes while providing a clean and extensible interface for future integration into larger applications.

# 2. API Overview

The API exposes two endpoints for service verification and purchase intent prediction.

| Method | Endpoint | Description |
|---------|----------|-------------|
| **GET** | `/` | Returns basic information about the prediction service and confirms that the API is running. |
| **POST** | `/predict` | Predicts whether an online shopping session will result in a purchase. |

All prediction requests are processed using the production model specified in the deployment configuration.

# 3. Authentication

The API does not implement authentication or authorization mechanisms, as it is designed for local deployment and educational purposes.

For production deployments, appropriate security measures such as API keys, OAuth 2.0, or reverse proxy authentication should be implemented to protect access to the prediction service.


# 4. API Endpoints

## 4.1 GET /

Returns basic information about the prediction service.

### Response

```json
{
    "service": "Shopper Intent Prediction API",
    "status": "healthy",
    "version": "1.0.0"
}
```

---

## 4.2 POST /predict

Predicts whether an online shopping session will result in a purchase.

### Request Body

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


# 5. Request and Response Schemas

The API validates all incoming requests using Pydantic models defined in `api/schemas.py`. Validation ensures that required fields are present and numerical values satisfy predefined constraints before prediction is performed.

## PredictionRequest

The request schema consists of the original shopping session attributes collected from the Online Shoppers Purchasing Intention dataset.

| Field | Type | Validation |
|--------|------|------------|
| Administrative | Integer | ≥ 0 |
| Administrative_Duration | Float | ≥ 0 |
| Informational | Integer | ≥ 0 |
| Informational_Duration | Float | ≥ 0 |
| ProductRelated | Integer | ≥ 0 |
| ProductRelated_Duration | Float | ≥ 0 |
| BounceRates | Float | ≥ 0 |
| ExitRates | Float | ≥ 0 |
| PageValues | Float | ≥ 0 |
| SpecialDay | Float | ≥ 0 |
| Month | String | Required |
| OperatingSystems | Integer | ≥ 1 |
| Browser | Integer | ≥ 1 |
| Region | Integer | ≥ 1 |
| TrafficType | Integer | ≥ 1 |
| VisitorType | String | Required |
| Weekend | Boolean | Required |

## PredictionResponse

| Field | Type | Description |
|--------|------|-------------|
| prediction | Integer | Binary prediction indicating whether the customer is expected to complete a purchase. |
| probability | Float | Predicted probability of purchase. |
| model | String | Name of the production model used for prediction. |
| model_version | String | Version of the deployed production model. |

# 6. Error Handling

FastAPI automatically validates incoming requests using the Pydantic schemas before invoking the prediction service.

The API may return the following HTTP status codes:

| Status Code | Description |
|------------|-------------|
| **200 OK** | Prediction completed successfully. |
| **422 Unprocessable Entity** | Request validation failed due to missing fields, invalid data types, or values outside the permitted range. |
| **500 Internal Server Error** | An unexpected error occurred while processing the prediction request. |

These validation mechanisms help ensure that only correctly formatted requests are passed to the machine learning pipeline.

# 7. Running the API

The prediction service can be started using Uvicorn from the project root directory.

```bash
uvicorn api.main:app --reload
```

Once running, the API is available at:

```
http://127.0.0.1:8000
```

FastAPI automatically generates interactive API documentation, allowing users to test endpoints directly from a web browser.

| URL | Description |
|-----|-------------|
| `/docs` | Interactive Swagger UI documentation. |
| `/redoc` | ReDoc API documentation. |

These interfaces provide a convenient way to inspect request schemas, execute prediction requests, and verify API responses without requiring external API clients.