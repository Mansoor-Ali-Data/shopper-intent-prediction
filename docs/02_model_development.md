# Model Development

## 1. Introduction

This document describes the development of the Shopper Intent Prediction model, covering the complete machine learning workflow from data preparation to production model selection.

The objective of the project is to predict whether an online shopping session will result in a purchase using historical browsing behaviour. Multiple machine learning algorithms were trained and evaluated to identify the model that best balances predictive performance and operational efficiency.

The development pipeline includes data preprocessing, feature engineering, handling class imbalance, model training, benchmarking, and production model selection. Rather than selecting a model based solely on predictive accuracy, the final production model is chosen using a weighted evaluation strategy that also considers inference latency and model size, making the solution more suitable for deployment.

## 2. Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset**, which contains anonymised web session information collected from an e-commerce platform. Each record represents a single user session and includes behavioural, administrative, and traffic-related attributes.

The target variable is **Revenue**, a binary label indicating whether the session resulted in a completed purchase.

The dataset contains a mixture of numerical and categorical features, including:

- Page visit statistics
- Time spent on different page categories
- Bounce and exit rates
- Visitor type
- Operating system and browser information
- Traffic source
- Month of visit
- Weekend indicator

Before model training, duplicate records are removed and the dataset is partitioned into training and testing subsets using a stratified train-test split to preserve the original class distribution.

## 3. Data Preprocessing

The data preprocessing stage transforms the raw online shopping session data into a numerical representation suitable for machine learning models. To ensure consistency between training and inference, all preprocessing operations are encapsulated within the `SessionPreprocessor` class, allowing the same fitted preprocessing pipeline to be reused after deployment. :contentReference[oaicite:0]{index=0}

The preprocessing workflow consists of the following steps:

1. **Duplicate Removal**

   Duplicate records are removed from the raw dataset before model development to eliminate redundant observations and reduce potential bias during training. :contentReference[oaicite:1]{index=1}

2. **Train-Test Split**

   The dataset is partitioned into training and testing subsets using a stratified train-test split. Stratification preserves the original class distribution of the target variable, ensuring both subsets remain representative of the overall dataset. :contentReference[oaicite:2]{index=2}

3. **Feature Transformation**

   Before encoding and scaling, the preprocessing pipeline invokes the feature engineering module to generate additional derived attributes from the raw session data. This guarantees identical feature generation during both training and inference. :contentReference[oaicite:3]{index=3}

4. **Categorical Encoding**

   Categorical variables are converted into numerical representations using one-hot encoding. The following categorical attributes are encoded:

   - Month
   - Visitor Type
   - Operating System
   - Browser
   - Region
   - Traffic Type

   Boolean variables such as `Weekend` and `Revenue` are converted into integer values for compatibility with machine learning algorithms. :contentReference[oaicite:4]{index=4}

5. **Feature Scaling**

   After encoding, numerical features are standardised using the `StandardScaler` implementation from Scikit-learn. Standardisation centres each feature around zero with unit variance, improving training stability for algorithms that are sensitive to feature magnitude. :contentReference[oaicite:5]{index=5}

6. **Schema Consistency**

   During training, the fitted feature schema is stored together with the preprocessing pipeline. During inference, incoming data are automatically aligned to the original feature schema, with any missing columns filled with zero values. This prevents feature mismatch errors when previously unseen categorical values are encountered. 

7. **Class Balancing**

   Following preprocessing, the training dataset is balanced using the Synthetic Minority Over-sampling Technique (SMOTE). This technique generates synthetic samples for the minority class, reducing class imbalance and improving the ability of the models to learn purchase behaviour. SMOTE is applied only to the training data after the train-test split to prevent data leakage. :contentReference[oaicite:7]{index=7}

## 4. Feature Engineering

Feature engineering was performed to enhance the predictive capability of the machine learning models by deriving additional behavioural attributes from the original online shopping session data. These engineered features capture user engagement and browsing behaviour more effectively than the raw variables alone, providing richer representations of customer interactions.

The feature engineering process is implemented in `features.py` and is executed automatically during both training and inference. Integrating feature generation into the preprocessing pipeline ensures that identical transformations are applied throughout the machine learning lifecycle, preventing inconsistencies between model training and deployment.

The following engineered features are created:

| Feature | Description | Formula |
|---------|-------------|---------|
| **Total_PageViews** | Total number of pages viewed during a session, representing browsing intensity. | `Administrative + Informational + ProductRelated` |
| **Total_Duration** | Total time spent across all page categories during the session. | `Administrative_Duration + Informational_Duration + ProductRelated_Duration` |
| **Avg_Time_Per_Page** | Average browsing time per viewed page, providing an indication of browsing depth. | `Total_Duration / (Total_PageViews + 1)` |
| **Engagement_Score** | Composite engagement metric combining browsing volume and session duration. Higher values indicate greater overall user engagement. | `Total_PageViews × Total_Duration` |
| **Product_Page_Ratio** | Proportion of visited pages that are product-related, reflecting the user's focus on product exploration. | `ProductRelated / (Total_PageViews + 1)` |

These engineered features are designed to represent different aspects of customer behaviour:

- **Browsing Intensity** is captured by **Total_PageViews**, indicating the overall number of pages visited during a session.
- **Session Engagement** is represented by **Total_Duration** and **Engagement_Score**, measuring the amount of time users spend interacting with the website.
- **Browsing Efficiency** is measured using **Avg_Time_Per_Page**, which distinguishes between quick navigation and more deliberate browsing behaviour.
- **Purchase Intent** is approximated by **Product_Page_Ratio**, as users who spend a greater proportion of their session viewing product pages are more likely to complete a purchase.

After feature generation, categorical variables are transformed using one-hot encoding, while boolean attributes are converted into numerical values. The resulting feature set is then passed to the preprocessing pipeline for scaling before being used for model training and inference.

By generating these additional behavioural features, the system provides machine learning models with richer information than the original dataset alone, improving their ability to distinguish between purchasing and non-purchasing sessions.

### Feature Consistency Between Training and Inference

A common challenge in production machine learning systems is maintaining consistency between the features used during model training (offline) and those generated during prediction (online). Differences between these two pipelines, commonly referred to as **training-serving skew**, can significantly degrade prediction quality.

This project avoids training-serving skew by using the same preprocessing pipeline for both training and inference. During model training, engineered features are generated before categorical encoding and feature scaling. The fitted preprocessing pipeline, including feature engineering, encoding mappings, scaling parameters, and feature schema, is then serialized and stored alongside the selected production model.

During inference, incoming prediction requests are processed using this same serialized preprocessing pipeline before being passed to the production model. As a result, both training and prediction use identical feature transformations, ensuring consistent model inputs throughout the machine learning lifecycle.

## 5. Model Development

To identify the most suitable production model, four classification algorithms were developed and evaluated. The selected models represent a combination of traditional machine learning techniques and deep learning, providing a balanced comparison between predictive performance and computational efficiency.

The models implemented in this project are:

| Model | Description |
|--------|-------------|
| **Logistic Regression** | A linear classification algorithm that serves as a strong baseline for binary classification problems. |
| **Decision Tree** | A tree-based classifier capable of learning non-linear decision boundaries through recursive feature partitioning. |
| **Random Forest** | An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. |
| **Artificial Neural Network (ANN)** | A feed-forward neural network designed to learn complex non-linear relationships within the shopping session data. |

The Artificial Neural Network consists of an input layer, two fully connected hidden layers with ReLU activation functions, and a sigmoid output layer for binary classification. The model is trained using the Adam optimizer and binary cross-entropy loss function, making it suitable for predicting purchase intent. Early stopping is employed during training to reduce overfitting by restoring the best-performing model based on validation loss. :contentReference[oaicite:0]{index=0}

To improve the robustness of the evaluation, all models are trained using the same preprocessed and balanced training dataset. This ensures that differences in performance are attributable to the learning algorithms rather than inconsistencies in the data preparation process.

Following training, each model is evaluated using both predictive and operational metrics. Predictive performance is measured using Accuracy, Precision, Recall, F1-score, and ROC-AUC, while operational performance is assessed through inference latency and serialized model size. These results are subsequently used during the production model selection process.

## 6. Model Configuration

The machine learning models were trained using predefined configurations rather than an automated hyperparameter optimisation process. This approach provides a consistent baseline for comparing different learning algorithms while keeping the focus of the project on the end-to-end machine learning engineering pipeline.

The classical machine learning models use their standard Scikit-learn implementations with a small number of manually specified parameters, such as the maximum number of iterations for Logistic Regression and the number of trees for Random Forest. The Artificial Neural Network is configured using the Adam optimizer, binary cross-entropy loss function, and a batch size and number of training epochs specified in the project configuration file.

Training parameters, including the number of epochs, batch size, and early stopping patience, are stored in `configs/training.yaml`, allowing model training behaviour to be modified without changing the source code. This configuration-driven approach improves reproducibility while simplifying experimentation.

## 7. Model Evaluation

Following model training, each candidate model is evaluated using a range of classification metrics to measure its ability to predict customer purchase intent. Because the problem is a binary classification task with an imbalanced class distribution, multiple evaluation metrics are considered instead of relying solely on overall accuracy.

The following evaluation metrics are used:

| Metric | Description |
|---------|-------------|
| **Accuracy** | Proportion of correctly classified shopping sessions. |
| **Precision** | Percentage of predicted purchases that are actual purchases. |
| **Recall** | Percentage of actual purchases correctly identified by the model. |
| **F1-Score** | Harmonic mean of Precision and Recall, providing a balanced measure of classification performance. |
| **ROC-AUC** | Measures the model's ability to distinguish between purchasing and non-purchasing sessions across different classification thresholds. |

These metrics provide complementary perspectives on model performance. While Accuracy measures overall prediction correctness, Precision and Recall evaluate the trade-off between false positives and false negatives. The F1-score is particularly important for this project because it balances both measures when working with an imbalanced dataset. ROC-AUC provides an overall assessment of the model's discrimination capability independent of the selected probability threshold.

The evaluation results for all candidate models are stored in `model_metrics.csv` and are later combined with engineering benchmark results during the production model selection process.

## 8. Benchmarking

In addition to predictive performance, the trained models are benchmarked using engineering metrics that reflect their suitability for deployment in a production environment. A model with the highest predictive accuracy may not always be the most practical choice if it exhibits slow inference performance or excessive storage requirements.

The benchmarking process evaluates each trained model using the following engineering metrics:

| Metric | Description |
|---------|-------------|
| **Average Inference Latency** | Average time required to generate a prediction for a single request. |
| **Serialized Model Size** | Storage size of the saved model artifact on disk. |

Inference latency is measured by repeatedly executing predictions on representative input samples after a number of warm-up runs. Multiple repetitions are performed to minimise the effect of execution variability and obtain a stable estimate of prediction performance.

The serialized model size is measured directly from the saved model artifact, providing an indication of storage requirements and deployment overhead. Smaller models are generally preferable for environments with limited computational or storage resources.

Benchmark results are recorded in `benchmark_metrics.csv` and combined with the predictive evaluation metrics during production model selection. By considering both predictive quality and operational efficiency, the system selects a model that is not only accurate but also practical for real-world deployment.

## 9. Production Model Selection

After all candidate models have been trained, evaluated, and benchmarked, the system automatically selects a production model using a weighted scoring approach. Rather than choosing the model with the highest predictive performance alone, the selection process considers both classification quality and deployment efficiency. This ensures that the selected model is suitable for real-world production environments where prediction accuracy, response time, and resource usage are all important.

Before calculating the overall score, metrics with different scales are normalised to ensure fair comparison across models. Predictive metrics such as F1-score and ROC-AUC are maximised, while engineering metrics including inference latency and model size are minimised. The normalised metrics are then combined into a weighted score, and the model with the highest overall score is selected as the production model.

The selected model, together with its preprocessing pipeline and deployment metadata, is recorded in `deployment.json`. This file stores the model version, artifact locations, and the reason for selection, allowing the prediction service to load the correct production model automatically.

### 9.1 Selection Metric Justification

The weighted scoring strategy is designed to balance predictive performance with operational requirements. Each evaluation metric contributes differently to the overall score according to its importance within the deployment pipeline.

| Metric | Weight | Justification |
|---------|--------|---------------|
| **F1-Score** | **0.45** | Assigned the highest weight because the dataset is imbalanced. F1-score balances Precision and Recall, providing a more reliable measure of classification performance than Accuracy alone. |
| **Inference Latency** | **0.30** | Low prediction latency is essential for responsive production systems. Prioritising latency helps ensure efficient real-time inference. |
| **ROC-AUC** | **0.15** | Measures the model's ability to distinguish between purchasing and non-purchasing sessions across different decision thresholds, providing an additional assessment of predictive quality. |
| **Model Size** | **0.10** | Smaller models require less storage and memory, making deployment more efficient. Since storage is generally less critical than prediction quality and inference speed, this metric receives the lowest weight. |

This weighting strategy reflects the objectives of a production-oriented machine learning system. F1-score receives the greatest importance because accurately identifying purchasing sessions is the primary objective of the project. Inference latency is also heavily weighted to ensure responsive prediction services, while ROC-AUC provides additional confidence in the model's discrimination capability. Model size is included as a secondary engineering consideration to encourage efficient deployment without sacrificing predictive performance.

By combining predictive and operational metrics into a single weighted score, the model selection process avoids favouring a model solely based on accuracy and instead identifies the most practical model for deployment.

## 10. Model Artifacts

Upon completion of the training, evaluation, benchmarking, and production model selection stages, the pipeline generates several artifacts required for deployment, inference, and system monitoring. These artifacts are stored in the `models/` directory and enable the prediction service to operate without retraining the models.

The generated artifacts are summarised below.

| Artifact | Description |
|----------|-------------|
| **Trained Model** | The selected production model serialized as a `.joblib` file for efficient loading during inference. |
| **Preprocessor** | Serialized preprocessing pipeline containing feature transformations, encoding mappings, scaling parameters, and feature schema used during training. |
| **deployment.json** | Stores deployment metadata, including the selected production model, artifact paths, model version, and selection rationale. |
| **model_metrics.csv** | Contains the predictive evaluation results for all candidate models, including Accuracy, Precision, Recall, F1-score, and ROC-AUC. |
| **benchmark_metrics.csv** | Records engineering benchmark results such as inference latency and serialized model size for each model. |
| **training_summary.md** | Provides a human-readable summary of the training process, evaluation results, benchmark outcomes, and the selected production model. |

Separating these artifacts from the source code improves reproducibility and simplifies deployment. During inference, the prediction service loads only the serialized production model, preprocessing pipeline, and deployment metadata, ensuring that incoming data undergoes the same transformations as the training data.

Storing evaluation and benchmark results as separate artifacts also improves traceability by preserving the performance of every candidate model. This enables future comparisons, model audits, and reproducible deployment decisions without requiring the models to be retrained.