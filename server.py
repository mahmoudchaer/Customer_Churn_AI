from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import threading
import time
import sys

app = Flask(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.best_run_id = None
        self.best_accuracy = 0
        self.model_accuracy = 0  # ✅ Stores current model accuracy

    def load_model(self):
        try:
            experiment = mlflow.get_experiment_by_name('Churn predictions')
            if experiment is None:
                raise Exception("Experiment 'Churn predictions' not found")

            runs = mlflow.search_runs([experiment.experiment_id])
            if runs.empty:
                raise Exception("No runs found in the experiment")

            best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
            best_accuracy = best_run['metrics.accuracy']

            # ✅ Just reload the model without restarting Flask
            if self.best_run_id != best_run.run_id and best_accuracy > self.best_accuracy:
                print(f"\n🚀 New model detected! Reloading... (Run ID: {best_run.run_id})")
                print(f"Previous Best Accuracy: {self.best_accuracy}")
                print(f"New Best Accuracy: {best_accuracy}")

                model_uri = f"runs:/{best_run.run_id}/model"  # Use "model" if that's how it's logged
                self.model = mlflow.sklearn.load_model(model_uri)
                self.best_run_id = best_run.run_id
                self.best_accuracy = best_accuracy
                self.model_accuracy = best_accuracy  # ✅ Store the accuracy
                print("✅ Model reloaded successfully!")

            else:
                print("🔄 No new model found. Keeping current model.")

        except Exception as e:
            print(f"⚠️ Error loading model: {str(e)}")

    
    def load_scaler(self):
        try:
            scaler = StandardScaler()
            df = pd.read_csv("processed_telco_data.csv")

            features_to_drop = [
                'gender', 'PhoneService', 'MultipleLines', 'PaymentMethod',
                'Partner', 'Dependents', 'PaperlessBilling', 'MonthlyCharges',
                'TotalCharges', 'Churn'
            ]
            X = df.drop(features_to_drop, axis=1)
            scaler.fit(X)
            self.scaler = scaler
            print("✅ Scaler loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading scaler: {str(e)}")

# Function to continuously check for new models
def monitor_mlflow():
    while True:
        model_manager.load_model()
        time.sleep(30)  # ✅ Check every 30 seconds

# Create global ModelManager instance
model_manager = ModelManager()
model_manager.load_model()
model_manager.load_scaler()

# Start MLflow model monitoring in a separate thread
threading.Thread(target=monitor_mlflow, daemon=True).start()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            if model_manager.model is None or model_manager.scaler is None:
                raise Exception("Model or scaler not available")
            
            input_data = {
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'tenure': int(request.form['tenure']),
                'InternetService': int(request.form['InternetService']),
                'OnlineSecurity': int(request.form['OnlineSecurity']),
                'OnlineBackup': int(request.form['OnlineBackup']),
                'DeviceProtection': int(request.form['DeviceProtection']),
                'TechSupport': int(request.form['TechSupport']),
                'StreamingTV': int(request.form['StreamingTV']),
                'StreamingMovies': int(request.form['StreamingMovies']),
                'Contract': int(request.form['Contract'])
            }

            input_df = pd.DataFrame([input_data])
            scaled_features = model_manager.scaler.transform(input_df)
            prediction = model_manager.model.predict(scaled_features)[0]
            prediction = "Yes" if prediction == 1 else "No"

        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction, model_accuracy=model_manager.model_accuracy)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .form-group { 
            margin-bottom: 15px; 
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        label { 
            display: inline-block; 
            width: 150px;
            font-weight: bold;
        }
        select, input { 
            width: 200px; 
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button { 
            padding: 10px 20px; 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            cursor: pointer;
            border-radius: 4px;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .prediction { 
            margin-top: 20px; 
            padding: 20px;
            background-color: #e8f5e9;
            border-radius: 5px;
            text-align: center;
        }
        .accuracy {
            margin-top: 20px;
            padding: 10px;
            background-color: #ffecb3;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        h1 {
            color: #2e7d32;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Customer Churn Prediction</h1>

    <div class="accuracy">
        <h2>Model Accuracy: {{ model_accuracy | round(4) }}</h2>
    </div>

    <form method="post">
        <div class="form-group">
            <label for="SeniorCitizen">Senior Citizen:</label>
            <select name="SeniorCitizen" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>

        <div class="form-group">
            <label for="tenure">Tenure (months):</label>
            <input type="number" name="tenure" required min="0">
        </div>

        <div class="form-group">
            <label for="InternetService">Internet Service:</label>
            <select name="InternetService" required>
                <option value="0">DSL</option>
                <option value="1">Fiber optic</option>
                <option value="2">No</option>
            </select>
        </div>

        <button type="submit">Predict Churn</button>
    </form>

    {% if prediction %}
    <div class="prediction">
        <h2>Prediction Result:</h2>
        <p>Will the customer churn? <strong>{{ prediction }}</strong></p>
    </div>
    {% endif %}
</body>
</html>
            ''')

    app.run(debug=True, use_reloader=False)
