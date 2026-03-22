import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("sales_model.pkl")

app = FastAPI(title="Sales Prediction API")


class PredictRequest(BaseModel):
    marketing_spend: float
    discount: float
    foot_traffic: float
    competitor_price: float
    day_of_week: int
    month: int


@app.post("/predict")
def predict(req: PredictRequest):
    features = np.array([[
        req.marketing_spend,
        req.discount,
        req.foot_traffic,
        req.competitor_price,
        req.day_of_week,
        req.month,
    ]])
    prediction = model.predict(features)[0]
    return {"predicted_sales": round(float(prediction), 2)}
