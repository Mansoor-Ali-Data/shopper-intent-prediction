# Training Summary

## Evaluation and Engineering Metrics

| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   ROC AUC |   Average Latency (ms) |   Model Size (MB) |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|-----------------------:|------------------:|
| random_forest       |     0.8947 |      0.6543 |   0.6937 |     0.6734 |    0.9193 |                42.6588 |           23.2971 |
| ann                 |     0.8681 |      0.5579 |   0.7565 |     0.6422 |    0.8999 |               100.7975 |            0.0600 |
| logistic_regression |     0.8595 |      0.5351 |   0.7775 |     0.6339 |    0.9088 |                 0.2112 |            0.0014 |
| decision_tree       |     0.8595 |      0.5423 |   0.6545 |     0.5931 |    0.7760 |                 0.1590 |            0.1621 |

## Production Model

Selected model: `random_forest`

Selected using weighted model selection (F1=45%, Latency=30%, ROC-AUC=15%, Model Size=10%). Overall Score = 0.7733.
