# 🎓 Student Performance Prediction using Machine Learning & FastAPI

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end supervised machine learning project and production REST API that predicts a student's **final academic exam score** (0–100) based on study habits, attendance records, past academic performance, assignment completion, and sleep quality.

---

## 📌 Table of Contents
- [Objective](#-objective)
- [Problem Statement & Framing](#-problem-statement--framing)
- [Dataset Description & Features](#-dataset-description--features)
- [Tech Stack](#-tech-stack)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Models Trained & Evaluation Metrics](#-models-trained--evaluation-metrics)
- [Results & Model Comparison](#-results--model-comparison)
- [Stage 8: FastAPI Backend Architecture](#-stage-8-fastapi-backend-architecture)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [API Usage & Examples](#-api-usage--examples)
- [Future Improvements](#-future-improvements)

---

## 🎯 Objective
To build, evaluate, and deploy a clean, interpretable regression model and production REST API that forecasts student exam performance early, enabling educators and academic advisors to provide targeted support to at-risk students before final examinations.

---

## 🧠 Problem Statement & Framing

### Why is this a **Regression** problem?
- **Target Variable (`final_score`)**: Continuous numerical value ranging from `0.0` to `100.0`.
- **Core Distinction**:
  - **Regression** predicts a *continuous quantity* ("What exact score will the student achieve?").
  - **Classification** predicts a *discrete category* ("Will the student Pass or Fail?").

---

## 📊 Dataset Description & Features

The project uses a dataset of **500 student records** (`data/student_performance.csv`).

### Feature Breakdown
| Feature Name | Role | Data Type | Range | Description |
|---|---|---|---|---|
| `study_hours` | **Feature (X)** | Continuous (float) | 0.5 – 10.0 | Daily hours spent studying |
| `attendance_pct` | **Feature (X)** | Continuous (float) | 50.0 – 100.0 | Percentage of classes attended |
| `prev_exam_score` | **Feature (X)** | Continuous (float) | 30.0 – 95.0 | Marks obtained in previous exam |
| `assignments_done` | **Feature (X)** | Discrete (int) | 0 – 10 | Total assignments submitted |
| `sleep_hours` | **Feature (X)** | Continuous (float) | 4.0 – 10.0 | Average nightly sleep hours |
| `final_score` | **Target (y)** | Continuous (float) | 0.0 – 100.0 | **Final exam score to predict** |

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn
- **Model Serialization**: Pickle
- **Backend API Framework**: FastAPI, Uvicorn, Pydantic

---

## 🔄 Machine Learning Workflow

```
[ 1. Dataset Generation & Loading ] ──► [ 2. Exploratory Data Analysis (EDA) ]
                                                        │
                                                        ▼
[ 4. Model Training & Evaluation  ] ◄── [ 3. Data Cleaning & X/y Split   ]
           │
           ▼
[ 5. Best Model Selection & Serialization ] ──► [ 6. Modular Prediction Pipeline ]
                                                        │
                                                        ▼
                                       [ 8. FastAPI REST API Backend     ]
```

1. **EDA**: Inspected distributions, calculated correlations, verified lack of nulls/duplicates, and plotted feature-target scatter plots.
2. **Data Cleaning**: Implemented median imputation for continuous variables, mode imputation for discrete variables, duplicate filtering, and IQR/domain-knowledge outlier clipping.
3. **Train/Test Split**: 80% Training set (400 students), 20% Holdout Test set (100 students) with `random_state=42`.
4. **Modelling**: Trained Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
5. **Evaluation**: Assessed models on Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
6. **Deployment**: Serialized the winning model to `src/best_model.pkl` and built a modular `predict_score()` function.
7. **FastAPI Backend**: Wrapped the prediction pipeline into a high-performance REST API with CORS support, Pydantic validation, and Swagger documentation.

---

## 🏆 Results & Model Comparison

Evaluated on the **100-student unseen test set**:

| Model | MAE ↓ | MSE ↓ | RMSE ↓ | $R^2$ ↑ | Selection Status |
|---|---|---|---|---|---|
| **Linear Regression** | **3.1023** | **16.7613** | **4.0941** | **0.8941** | 🥇 **Selected Model** |
| **Random Forest Regressor** | 3.6559 | 21.8468 | 4.6741 | 0.8620 | 🥈 Runner-Up |
| **Decision Tree Regressor** | 5.6484 | 52.6363 | 7.2551 | 0.6674 | 🥉 Overfit |

---

## ⚡ Stage 8: FastAPI Backend Architecture

The backend (`backend/main.py`) provides a production-grade interface to the ML pipeline:

### Key Features
- **Pydantic Validation (`StudentInput`)**: Enforces range checks (e.g. `0.0 <= study_hours <= 24.0`, `0 <= assignments_done <= 10`).
- **Interactive Swagger Documentation**: Available at `http://127.0.0.1:8000/docs`.
- **ReDoc Documentation**: Available at `http://127.0.0.1:8000/redoc`.
- **CORS Support**: Enabled for cross-origin integration with web apps.

### API Endpoints
| Method | Endpoint | Description | Sample Status |
|---|---|---|---|
| `GET` | `/` | API Welcome message & endpoint directory | 200 OK |
| `GET` | `/health` | Service health status & model loading verification | 200 OK |
| `POST` | `/predict` | Predict final score for a single student profile | 200 OK / 422 Unprocessable |
| `POST` | `/predict/batch` | Predict final scores for multiple student profiles | 200 OK / 422 Unprocessable |

---

## 📁 Repository Structure

```
student-performance-prediction-ml/
│
├── backend/
│   ├── main.py                       # FastAPI application & endpoints
│   └── test_api.py                   # Automated API integration test script
├── data/
│   ├── generate_dataset.py           # Synthetic dataset generator script
│   ├── student_performance.csv       # Raw dataset (500 records)
│   └── student_performance_cleaned.csv # Cleaned dataset after Stage 3
├── notebooks/
│   ├── student_performance_prediction.ipynb # Interactive Jupyter Notebook (64 cells)
│   ├── eda_stage2.py                 # EDA execution script
│   ├── stage3_cleaning_framing.py    # Data cleaning & problem framing script
│   ├── stage4_modeling.py            # Model training & comparison script
│   └── stage5_prediction.py          # Best model selection & prediction demo
├── src/
│   ├── best_model.pkl                # Serialized winning model (Linear Regression)
│   ├── feature_names.pkl             # Serialized feature list
│   └── prediction.py                 # Production prediction module
├── README.md                         # Complete project documentation
├── requirements.txt                  # Python dependency requirements
└── .gitignore                        # Git exclusion rules
```

---

## 🚀 How to Run

### 1. Setup Environment & Dependencies
```bash
git clone https://github.com/your-username/student-performance-prediction-ml.git
cd student-performance-prediction-ml
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to test endpoints interactively!

### 3. Run Automated API Tests
```bash
python backend/test_api.py
```

---

## 💻 API Usage & Examples

### Single Prediction Request (`POST /predict`)
```json
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "study_hours": 8.0,
  "attendance_pct": 90.0,
  "prev_exam_score": 85.0,
  "assignments_done": 9,
  "sleep_hours": 7.5
}
```

### Single Prediction Response
```json
{
  "status": "success",
  "predicted_score": 97.7,
  "grade": "A",
  "interpretation": "Outstanding performance expected!",
  "inputs": {
    "study_hours": 8.0,
    "attendance_pct": 90.0,
    "prev_exam_score": 85.0,
    "assignments_done": 9,
    "sleep_hours": 7.5
  }
}
```

---

## 🔮 Future Improvements
1. **Frontend Integration**: Connect this FastAPI backend to a React or Streamlit user dashboard.
2. **Containerization**: Package the FastAPI server into a Docker image for cloud deployment (AWS ECS / GCP Cloud Run).
3. **Database Integration**: Log prediction requests to PostgreSQL for model drift monitoring.
