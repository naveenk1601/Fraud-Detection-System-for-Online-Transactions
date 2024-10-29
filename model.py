import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load dataset
data = pd.read_csv('fraud_data.csv')

# Preprocess the data
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'fraud_detection_model.joblib')

# Function to predict
def predict_fraud(input_data):
    model = joblib.load('fraud_detection_model.joblib')
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]
