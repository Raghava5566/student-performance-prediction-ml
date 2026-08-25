"""
main.py
=======
FastAPI REST API Backend for Student Performance Prediction.

Provides HTTP endpoints:
  - GET  /         --> Root welcome message & documentation pointers
  - GET  /health   --> API & model health status
  - POST /predict  --> Predict score for a single student (JSON payload)
  - POST /predict/batch --> Predict scores for multiple students (JSON payload)

Features:
  - Pydantic schema validation (field limits, descriptions, example values)
  - CORS middleware enabled
  - Auto-generated Swagger documentation at /docs and ReDoc at /redoc
"""

import sys
import os
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in Python path so we can import src.prediction
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import predict_score, load_model, FEATURE_COLUMNS, MODEL_PATH

# ── Global Model Holder ───────────────────────────────────────────────────────
loaded_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: Loads trained ML model when FastAPI server starts up.
    """
    global loaded_model
    try:
        loaded_model = load_model(MODEL_PATH)
        print(f"[OK] FastAPI Startup: Successfully loaded model from '{MODEL_PATH}'")
    except Exception as e:
        print(f"[WARNING] FastAPI Startup Warning: Could not load model: {e}")
        loaded_model = None
    yield
    print("[STOP] FastAPI Shutdown complete.")


# ── FastAPI App Instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "Production REST API for predicting student final exam scores using "
        "supervised Machine Learning (Linear Regression). Built with FastAPI, Pydantic, "
        "and Scikit-Learn."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS Middleware Configuration ─────────────────────────────────────────────
# Allows web frontends (React, Vue, HTML/JS) hosted on any origin to query this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request Schemas ──────────────────────────────────────────────────
class StudentInput(BaseModel):
    """
    Input schema for a single student profile with strict range validation.
    """
    study_hours: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Daily study hours (0.0 to 24.0)",
        json_schema_extra={"example": 7.5}
    )
    attendance_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of classes attended (0.0 to 100.0)",
        json_schema_extra={"example": 88.0}
    )
    prev_exam_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score in the previous exam (0.0 to 100.0)",
        json_schema_extra={"example": 78.0}
    )
    assignments_done: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of assignments submitted (0 to 10)",
        json_schema_extra={"example": 8}
    )
    sleep_hours: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Average nightly sleep hours (0.0 to 24.0)",
        json_schema_extra={"example": 7.0}
    )


class BatchStudentInput(BaseModel):
    """
    Input schema for batch predictions (list of student profiles).
    """
    students: List[StudentInput] = Field(
        ...,
        description="List of student profiles for batch prediction",
        json_schema_extra={
            "example": [
                {
                    "study_hours": 9.0,
                    "attendance_pct": 95.0,
                    "prev_exam_score": 88.0,
                    "assignments_done": 10,
                    "sleep_hours": 7.5
                },
                {
                    "study_hours": 3.0,
                    "attendance_pct": 60.0,
                    "prev_exam_score": 45.0,
                    "assignments_done": 2,
                    "sleep_hours": 5.5
                }
            ]
        }
    )


# ── Pydantic Response Schemas ─────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    status: str = Field(default="success", description="Response status")
    predicted_score: float = Field(..., description="Predicted final exam score (0.0 to 100.0)")
    grade: str = Field(..., description="Letter grade equivalent (A, B, C, D, F)")
    interpretation: str = Field(..., description="Qualitative feedback summary")
    inputs: dict = Field(..., description="Mirror of input parameters evaluated")


class BatchPredictionResponse(BaseModel):
    status: str = Field(default="success", description="Response status")
    total_students: int = Field(..., description="Count of evaluated profiles")
    predictions: List[PredictionResponse] = Field(..., description="Individual student prediction reports")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service operational status ('healthy' or 'degraded')")
    model_loaded: bool = Field(..., description="True if trained ML model is loaded in memory")
    model_name: str = Field(..., description="Name of the active ML algorithm")
    features_expected: List[str] = Field(..., description="List of expected feature column names")
    api_version: str = Field(default="1.0.0")


# ── Helper Function ───────────────────────────────────────────────────────────
def get_grade_and_interpretation(score: float):
    """Returns letter grade and human-readable feedback based on score."""
    if score >= 90.0:
        return "A", "Outstanding performance expected!"
    elif score >= 75.0:
        return "B", "Good performance expected."
    elif score >= 60.0:
        return "C", "Average performance expected."
    elif score >= 45.0:
        return "D", "Below average performance — extra study recommended."
    else:
        return "F", "At risk — urgent academic support recommended."


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", summary="Root Endpoint", tags=["General"])
def read_root():
    """
    Root welcome endpoint with service links.
    """
    return {
        "message": "Welcome to the Student Performance Prediction API!",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict"
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check", tags=["General"])
def health_check():
    """
    Verifies service health and checks if the trained ML model is active.
    """
    is_loaded = loaded_model is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=type(loaded_model).__name__ if is_loaded else "None",
        features_expected=FEATURE_COLUMNS,
        api_version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Single Student Score", tags=["Prediction"])
def predict_single(data: StudentInput):
    """
    Accepts student study and attendance attributes, predicts final exam score,
    and returns letter grade plus qualitative evaluation.
    """
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained ML model is not loaded. Ensure src/best_model.pkl exists."
        )
    
    try:
        score = predict_score(
            study_hours=data.study_hours,
            attendance_pct=data.attendance_pct,
            prev_exam_score=data.prev_exam_score,
            assignments_done=data.assignments_done,
            sleep_hours=data.sleep_hours,
            model=loaded_model,
            verbose=False
        )
        score_rounded = round(score, 1)
        grade, interp = get_grade_and_interpretation(score_rounded)

        return PredictionResponse(
            status="success",
            predicted_score=score_rounded,
            grade=grade,
            interpretation=interp,
            inputs=data.model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Predict Batch of Student Scores", tags=["Prediction"])
def predict_batch_api(data: BatchStudentInput):
    """
    Accepts a list of student profiles and returns predictions for all of them.
    """
    if loaded_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained ML model is not loaded. Ensure src/best_model.pkl exists."
        )
    
    results = []
    for student in data.students:
        score = predict_score(
            study_hours=student.study_hours,
            attendance_pct=student.attendance_pct,
            prev_exam_score=student.prev_exam_score,
            assignments_done=student.assignments_done,
            sleep_hours=student.sleep_hours,
            model=loaded_model,
            verbose=False
        )
        score_rounded = round(score, 1)
        grade, interp = get_grade_and_interpretation(score_rounded)

        results.append(
            PredictionResponse(
                status="success",
                predicted_score=score_rounded,
                grade=grade,
                interpretation=interp,
                inputs=student.model_dump()
            )
        )

    return BatchPredictionResponse(
        status="success",
        total_students=len(results),
        predictions=results
    )


# ── Server Execution ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
