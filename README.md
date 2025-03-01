# Customer Churn Prediction with MLflow

## Mahmoud Chaer  
**EECE 503N**  
**Assignment: MLflow Experiment Tracking and Model Registry**

## Introduction
This project focuses on predicting customer churn using a logistic regression model. MLflow was used for experiment tracking across four runs with different train-test splits, hyperparameters, and regularization techniques. The best model was deployed using a Flask web server.
# Running the Server

To run the Flask web server, use the following command:

```bash
python server.py
```

This will start the server and provide real-time predictions using the best model registered in MLflow. The server will check for better models in MLflow every 30 seconds.

## Dataset & Preprocessing
- Dataset: **WA_Fn-UseC_-Telco-Customer-Churn.csv**
- Feature selection: Removed weakly correlated features (e.g., Gender, PhoneService, PaymentMethod, etc.)
- Data splitting:
  - Run 1: 80-20 split (Baseline Model)
  - Run 2: 70-30 split (Hyperparameter Tuning)
  - Run 3: 80-20 split (Feature Selection)
  - Run 4: 75-25 split (L1 Regularization)
- StandardScaler applied for normalization

## Experiment Results
| Run | Split | Hyperparameters | Feature Selection | Regularization | Accuracy |
|----|------|---------------|----------------|--------------|----------|
| 1  | 80-20 | Default       | No             | No           | **81.33%** |
| 2  | 70-30 | C=0.8, max_iter=300 | No | No | 80.45%  |
| 3  | 80-20 | Default       | Yes            | No           | 79.41%  |
| 4  | 75-25 | C=0.5         | No             | L1 (Lasso)   | 79.72%  |

## MLflow & Deployment
- MLflow tracked all runs and logged metrics.
- Best model (**Run 1**) registered in **MLflow Model Registry**.
- Flask web server deployed to provide real-time predictions.
- Server checks for better models in MLflow every **30 seconds**.

## Conclusion
- MLflow enabled efficient experiment tracking.
- Hyperparameter tuning and feature selection impacted model performance.
- Flask deployment ensures the best model is always in production.

## References
- [MLflow](https://mlflow.org)
- [Scikit-learn](https://scikit-learn.org)
- [Flask](https://flask.palletsprojects.com/)


