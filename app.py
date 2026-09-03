from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle


# ==========================================
# 1. Create FastAPI Application
# ==========================================

app = FastAPI(
    title="AI Financial Controller API",
    description="Employee CTC Anomaly Detection API",
    version="1.0.0"
)


# ==========================================
# 2. CORS Configuration
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. Load ML Model
# ==========================================

with open("salary_fit_model.pkl", "rb") as file:
    model_data = pickle.load(file)


model = model_data["model"]
role_encoder = model_data["role_encoder"]
target_encoder = model_data["target_encoder"]


# ==========================================
# 4. Input Data Structure
# ==========================================

class EmployeeData(BaseModel):

    experience: float
    role: str
    ctc: float


# ==========================================
# 5. Home / Health Check
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Financial Controller API is running",
        "status": "success"
    }


# ==========================================
# 6. Get Available Roles
# ==========================================

@app.get("/roles")
def get_roles():

    return {
        "roles": role_encoder.classes_.tolist()
    }


# ==========================================
# 7. Prediction API
# ==========================================

@app.post("/predict")
def predict_employee(data: EmployeeData):

    # --------------------------------------
    # Convert Role into Role Code
    # --------------------------------------

    role_code = role_encoder.transform(
        [data.role]
    )[0]


    # --------------------------------------
    # Create Input DataFrame
    # --------------------------------------

    new_employee = pd.DataFrame({

        "Years of Experience": [
            data.experience
        ],

        "Current Salary": [
            data.ctc
        ],

        "Role_Code": [
            role_code
        ]
    })


    # --------------------------------------
    # Make Prediction
    # --------------------------------------

    prediction = model.predict(
        new_employee
    )


    # --------------------------------------
    # Convert Prediction to Original Label
    # --------------------------------------

    result = target_encoder.inverse_transform(
        prediction
    )[0]


    # ======================================
    # Determine Anomaly Status
    # ======================================

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


    # ======================================
    # Return Result
    # ======================================

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