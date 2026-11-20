from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Banking Campaign Response API")

# Load saved model and preprocessing tools
artifacts = joblib.load("model_artifacts.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
numeric_features = artifacts["numeric_features"]
encoded_columns = artifacts["encoded_columns"]

# Input schema (excluding 'duration' and 'y' as per your notebook)
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def home():
    return {"message": "Banking Campaign Prediction API is running!"}

@app.post("/predict")
def predict_subscription(data: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # One-Hot Encoding matching training columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)
    
    # Scale numeric features
    input_encoded[numeric_features] = scaler.transform(input_encoded[numeric_features])
    
    # Predict class and probability
    prediction = int(model.predict(input_encoded)[0])
    probability = float(model.predict_proba(input_encoded)[0][1])
    
    return {
        "will_subscribe": "Yes" if prediction == 1 else "No",
        "subscription_probability": round(probability, 4)
    }