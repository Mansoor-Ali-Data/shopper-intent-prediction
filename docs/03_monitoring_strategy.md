# 1. Introduction

Machine learning systems require continuous monitoring after deployment to ensure that prediction quality remains reliable as new data is received. Changes in data distributions, missing values, or degraded data quality may reduce model performance over time, even if the model performed well during initial evaluation.

This project incorporates a lightweight monitoring module that evaluates incoming datasets for potential data quality issues and feature distribution drift. Rather than automatically retraining the model, the monitoring component analyses the collected statistics and determines whether retraining should be recommended based on predefined thresholds.

The monitoring workflow is implemented in `monitoring.py` and produces structured reports that summarise the current health of the deployed machine learning system. These reports support informed maintenance decisions while keeping the deployment pipeline simple and reproducible.

# 2. Monitoring Objectives

The monitoring component is designed to ensure that the deployed machine learning model continues to receive data that is consistent with the conditions under which it was trained. Monitoring focuses on identifying potential issues that could reduce prediction reliability before they significantly impact production performance.

The monitoring strategy has four primary objectives:

- Detect missing values that may indicate data quality problems.
- Identify changes in feature distributions between historical and newly observed data.
- Estimate the overall proportion of drifting features.
- Recommend model retraining when predefined monitoring thresholds are exceeded.

Unlike automated retraining systems, this project provides recommendations rather than immediately replacing the deployed model. This approach allows model updates to remain under human supervision while still providing early warning of potential performance degradation.

# 3. Data Quality Monitoring

Before analysing feature drift, the monitoring system evaluates the quality of incoming datasets by checking for missing values within each feature. Missing values may indicate data collection errors, schema changes, or incomplete preprocessing that could affect prediction quality.

For every feature, the system calculates the percentage of missing values relative to the total number of observations. These statistics are included in the monitoring report and provide an overview of dataset completeness.

Monitoring missing values helps identify potential data pipeline issues before they propagate to the prediction service, reducing the risk of unreliable model predictions.

# 4. Feature Drift Detection

The monitoring module compares the statistical distribution of incoming data against the reference training dataset to identify feature drift. Drift occurs when the characteristics of production data differ significantly from the data used during model training, potentially reducing predictive performance.

Each feature is analysed individually by comparing its statistical properties between the reference and current datasets. Features whose differences exceed the predefined drift threshold are flagged as drifting.

The monitoring report records both the number and proportion of drifting features, providing an overall indication of dataset stability. This information allows changes in customer behaviour or data collection processes to be detected before significant degradation in model performance occurs.

# 5. Retraining Recommendation

Rather than automatically retraining the deployed model, the monitoring system evaluates whether retraining is likely to be beneficial based on the observed monitoring statistics.

A retraining recommendation is generated when one or more monitoring thresholds are exceeded, including excessive feature drift or unacceptable levels of missing data. This rule-based approach provides a simple and transparent decision-making process while avoiding unnecessary retraining.

By separating monitoring from model retraining, the system maintains greater operational control and allows new models to be validated before deployment.

# 6. Incident Response Scenario

When the monitoring system detects excessive feature drift or poor data quality, it generates a retraining recommendation rather than automatically replacing the deployed model.

The incident response process consists of the following steps:

1. The monitoring module analyses newly collected production data.
2. Missing value statistics and feature drift metrics are calculated.
3. If predefined thresholds are exceeded, the monitoring report recommends model retraining.
4. Engineers review the monitoring report to determine whether the detected changes are significant.
5. If retraining is required, the training pipeline is executed using updated data to produce new candidate models.
6. Candidate models are evaluated, benchmarked, and compared using the weighted production model selection process.
7. If a newly trained model demonstrates improved overall performance, it replaces the existing production model and the deployment metadata is updated.

This workflow ensures that model updates remain controlled and reproducible while reducing the risk of deploying poorly performing models without human verification.

# 7. Monitoring Reports

The monitoring process generates a structured report containing the results of each monitoring run. This report is stored as `monitoring_report.json` within the project reports directory.

The report includes information such as:

- Missing value statistics
- Feature drift analysis
- Percentage of drifting features
- Retraining recommendation
- Overall monitoring summary

Persisting monitoring results enables historical comparisons and provides an audit trail for model maintenance activities.

# 8. Limitations

The implemented monitoring strategy provides a lightweight approach suitable for demonstrating production monitoring concepts. However, several limitations remain.

The system focuses primarily on data quality and feature distribution drift rather than directly measuring prediction quality after deployment. It also uses threshold-based decision rules instead of more advanced statistical monitoring techniques.

Future improvements could include concept drift detection, automated performance monitoring using labelled production data, scheduled retraining pipelines, and integration with external monitoring platforms for continuous model health tracking.
