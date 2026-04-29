from fastapi import FastAPI
import numpy as np
import joblib
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from app.schemas import InputData
from fastapi.responses import Response

app = FastAPI()

# Load models
binary_model = CatBoostClassifier()
binary_model.load_model("models/binary_model.cbm")

multi_model = XGBClassifier()
multi_model.load_model("models/multi_model.json")

scaler = joblib.load("models/scaler.pkl")

custom_mapping = {
    0: 'No Failure',
    1: 'Overstrain Failure',
    2: 'Power Failure',
    3: 'Heat Dissipation Failure',
    4: 'Tool Wear Failure',
    5: 'Random Failures'
}

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API running"}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    pred_binary = binary_model.predict(input_scaled)

    if pred_binary[0] == 1:
        pred_multi = multi_model.predict(input_scaled)
        return {
            "failure": True,
            "type": custom_mapping[int(pred_multi[0])]
        }
    else:
        return {"failure": False}