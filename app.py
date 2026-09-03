import os
import pickle
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================
# Create App
# ==========================================

app = FastAPI(
    title="AI Financial Controller API"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# Load Model
# ==========================================

with open("salary_fit_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
role_encoder = model_data["role_encoder"]
target_encoder = model_data["target_encoder"]


# ==========================================
# Input Structure
# ==========================================

class EmployeeData(BaseModel):
    experience: float
    role: str
    ctc: float


# ==========================================
# Health Check
# ==========================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "AI Financial Controller API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==========================================
# Get Roles
# ==========================================

@app.get("/roles")
def get_roles():

    return {
        "roles": role_encoder.classes_.tolist()
    }


# ==========================================
# Prediction
# ==========================================

@app.post("/predict")
def predict_employee(data: EmployeeData):

    # Convert role to number
    role_code = role_encoder.transform(
        [data.role]
    )[0]

    # Create model input
    new_employee = pd.DataFrame({
        "Years of Experience": [data.experience],
        "Current Salary": [data.ctc],
        "Role_Code": [role_code]
    })

    # Prediction
    prediction = model.predict(new_employee)

    # Convert prediction to text
    result = target_encoder.inverse_transform(
        prediction
    )[0]

    # Determine status
    if result == "Good Fit":

        anomaly_status = "Not Anomaly"
        ctc_status = "Good Fit"

        message = (
            "The employee's CTC is appropriate "
            "for their role and experience."
        )

    elif result == "Bad Fit - Underpaid":

        anomaly_status = "Anomaly"
        ctc_status = "Underpaid"

        message = (
            "The employee's CTC is lower than expected "
            "for their role and experience."
        )

    elif result == "Bad Fit - Overpaid":

        anomaly_status = "Anomaly"
        ctc_status = "Overpaid"

        message = (
            "The employee's CTC is higher than expected "
            "for their role and experience."
        )

    else:

        anomaly_status = "Unknown"
        ctc_status = result
        message = "Unable to determine CTC status."

    return {
        "prediction": result,
        "anomaly_status": anomaly_status,
        "ctc_status": ctc_status,
        "message": message,
        "employee": {
            "experience": data.experience,
            "role": data.role,
            "ctc": data.ctc
        }
    }


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 10000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )