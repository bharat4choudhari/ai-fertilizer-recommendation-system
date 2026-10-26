from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('saved_model.pkl')

class SensorData(BaseModel):
    Temperature: float
    Moisture: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    PH: float
    Soil: str
    Crop: str

@app.post("/predict")
def predict_fertilizer(data: SensorData):
    input_df = pd.DataFrame([{
        'Temperature': data.Temperature,
        'Moisture': data.Moisture,
        'Nitrogen': data.Nitrogen,
        'Phosphorous': data.Phosphorous,
        'Potassium': data.Potassium,
        'PH': data.PH,
        'Soil': data.Soil,
        'Crop': data.Crop
    }])
    
    prediction = model.predict(input_df)
    
    return {
        "status": "success",
        "predicted_fertilizer": prediction[0]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)