"""
prediction.py
=============
Reusable prediction module for the Student Performance Prediction project.

This is the PRODUCTION-READY piece of the project — a clean, importable
module that any Python script or web app can call to get a score prediction.

Usage (from project root):
    from src.prediction import predict_score, load_model, train_and_save_model

Quick example:
    score = predict_score(
        study_hours=7.0,
        attendance_pct=85.0,
        prev_exam_score=72.0,
        assignments_done=8,
        sleep_hours=7.5
    )
    print(f"Predicted score: {score}")
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Third-party Libraries ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# ── Constants ─────────────────────────────────────────────────────────────────
# These paths are relative to the project ROOT directory.
# Always run scripts from the project root (student-performance-prediction-ml/)

DATA_PATH    = os.path.join("data", "student_performance.csv")
MODEL_PATH   = os.path.join("src",  "best_model.pkl")
FEATURES_PATH= os.path.join("src",  "feature_names.pkl")

# The feature columns our model expects — ORDER MATTERS.
# The model was trained with these exact columns in this exact order.
# Changing this order would give wrong predictions.
FEATURE_COLUMNS = [
    "study_hours",       # continuous: daily study hours
    "attendance_pct",    # continuous: % classes attended
    "prev_exam_score",   # continuous: previous exam score
    "assignments_done",  # discrete:   assignments submitted (0-10)
    "sleep_hours",       # continuous: average nightly sleep hours
]
TARGET_COLUMN = "final_score"

# Valid input ranges for each feature (used in input validation)
VALID_RANGES = {
    "study_hours":       (0.0,  24.0),
    "attendance_pct":    (0.0, 100.0),
    "prev_exam_score":   (0.0, 100.0),
    "assignments_done":  (0,    10),
    "sleep_hours":       (0.0,  24.0),
}


# =============================================================================
# FUNCTION 1: train_and_save_model
# =============================================================================
def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH,
                         features_path=FEATURES_PATH, random_state=42):
    """
    Train all three models, select the best one by R², and save it to disk.

    This function is meant to be run ONCE (or when you retrain the model).
    After running this, the saved model can be loaded instantly for predictions.

    Parameters:
    -----------
    data_path     : str  — Path to the CSV dataset
    model_path    : str  — Where to save the trained model (.pkl)
    features_path : str  — Where to save the feature name list (.pkl)
    random_state  : int  — Seed for reproducibility

    Returns:
    --------
    dict  — Training results including best model name, R², RMSE
    """
    # ── Load Data ─────────────────────────────────────────────────────────────
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            f"Make sure you run this from the project root directory."
        )
    df = pd.read_csv(data_path)
    X  = df[FEATURE_COLUMNS]
    y  = df[TARGET_COLUMN]

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # ── Train All Three Candidates ────────────────────────────────────────────
    candidates = {
        "Linear Regression":  LinearRegression(),
        "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest":      RandomForestRegressor(
                                  n_estimators=100, max_depth=10,
                                  random_state=random_state
                              ),
    }

    eval_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        eval_results[name] = {
            "model": model,
            "R2":    r2_score(y_test, preds),
            "RMSE":  float(np.sqrt(mean_squared_error(y_test, preds))),
            "MAE":   float(mean_absolute_error(y_test, preds)),
        }

    # ── Select Best by R² ─────────────────────────────────────────────────────
    best_name  = max(eval_results, key=lambda n: eval_results[n]["R2"])
    best_entry = eval_results[best_name]
    best_model = best_entry["model"]

    # ── Persist to Disk ───────────────────────────────────────────────────────
    # Ensure the src/ directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(features_path, "wb") as f:
        pickle.dump(FEATURE_COLUMNS, f)

    return {
        "best_model_name": best_name,
        "best_model":      best_model,
        "R2":              best_entry["R2"],
        "RMSE":            best_entry["RMSE"],
        "MAE":             best_entry["MAE"],
        "all_results":     {
            k: {m: round(v, 4) for m, v in vd.items() if m != "model"}
            for k, vd in eval_results.items()
        },
    }


# =============================================================================
# FUNCTION 2: load_model
# =============================================================================
def load_model(model_path=MODEL_PATH):
    """
    Load the saved trained model from disk.

    Call this once at the start of your program and reuse the model object.
    Loading from disk is fast — much faster than retraining.

    Parameters:
    -----------
    model_path : str — Path to the .pkl file

    Returns:
    --------
    trained sklearn model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at '{model_path}'.\n"
            f"Run train_and_save_model() first to create it."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# =============================================================================
# FUNCTION 3: predict_score  (THE MAIN PREDICTION FUNCTION)
# =============================================================================
def predict_score(study_hours, attendance_pct, prev_exam_score,
                  assignments_done, sleep_hours,
                  model=None, verbose=False):
    """
    Predict the final exam score for a single student.

    This is the function you'll call in interviews, demos, and applications.
    It handles everything: validation, formatting, prediction, and clipping.

    Parameters:
    -----------
    study_hours      : float  — Daily study hours  (e.g., 6.5)
    attendance_pct   : float  — Attendance %        (e.g., 85.0)
    prev_exam_score  : float  — Previous exam score (e.g., 72.0)
    assignments_done : int    — Assignments done    (e.g., 8)
    sleep_hours      : float  — Nightly sleep hours (e.g., 7.0)
    model            : sklearn model (optional — loads from disk if None)
    verbose          : bool   — Print a detailed breakdown (default False)

    Returns:
    --------
    float — Predicted final score, clipped to [0, 100]

    Raises:
    -------
    ValueError if any input is outside its valid range
    FileNotFoundError if model hasn't been saved yet
    """
    # ── Step 1: Load model (lazy loading — only if not passed in) ─────────────
    if model is None:
        model = load_model()

    # ── Step 2: Input Validation ──────────────────────────────────────────────
    inputs = {
        "study_hours":       study_hours,
        "attendance_pct":    attendance_pct,
        "prev_exam_score":   prev_exam_score,
        "assignments_done":  assignments_done,
        "sleep_hours":       sleep_hours,
    }
    errors = []
    for feature, value in inputs.items():
        lo, hi = VALID_RANGES[feature]
        if not (lo <= value <= hi):
            errors.append(
                f"  '{feature}' = {value} is outside valid range [{lo}, {hi}]"
            )
    if errors:
        raise ValueError(
            "Input validation failed:\n" + "\n".join(errors) + "\n"
            "Please provide values within the valid ranges shown above."
        )

    # ── Step 3: Build Input DataFrame ─────────────────────────────────────────
    # We wrap the inputs in a DataFrame so the model receives the same
    # structure it was trained on — same column names, same order.
    # A bare list or array would work too, but a named DataFrame is safer
    # (it explicitly maps each value to the right feature).
    input_df = pd.DataFrame([{
        "study_hours":       float(study_hours),
        "attendance_pct":    float(attendance_pct),
        "prev_exam_score":   float(prev_exam_score),
        "assignments_done":  float(assignments_done),
        "sleep_hours":       float(sleep_hours),
    }])

    # ── Step 4: Generate Prediction ───────────────────────────────────────────
    # model.predict() always returns a numpy array, even for one sample.
    # We take [0] to extract the single scalar value.
    raw = model.predict(input_df)[0]

    # ── Step 5: Clip to Valid Output Range ────────────────────────────────────
    # A purely mathematical model can technically output values like 101 or -2.
    # clip() ensures we never return an impossible score.
    predicted = float(np.clip(raw, 0.0, 100.0))

    # ── Step 6: Verbose Output ────────────────────────────────────────────────
    if verbose:
        _print_prediction_report(inputs, raw, predicted)

    return predicted


# =============================================================================
# FUNCTION 4: predict_batch  (predict for many students at once)
# =============================================================================
def predict_batch(df_input, model=None):
    """
    Predict scores for multiple students from a DataFrame.

    Parameters:
    -----------
    df_input : pd.DataFrame — Must contain all 5 feature columns
    model    : sklearn model (optional — loads from disk if None)

    Returns:
    --------
    pd.Series — Predicted scores (clipped to [0, 100])
    """
    if model is None:
        model = load_model()

    # Check all required columns are present
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing columns: {missing_cols}")

    preds = model.predict(df_input[FEATURE_COLUMNS])
    return pd.Series(np.clip(preds, 0, 100), index=df_input.index,
                     name="predicted_final_score")


# =============================================================================
# HELPER: pretty print report
# =============================================================================
def _print_prediction_report(inputs, raw, predicted):
    """Internal helper — prints a nicely formatted prediction report."""
    grades = [(90, "A", "Outstanding!"), (75, "B", "Good performance."),
              (60, "C", "Average."),     (45, "D", "Below average."),
              (0,  "F", "At risk.")]
    grade, msg = next(
        (g, m) for threshold, g, m in grades if predicted >= threshold
    )

    print("\n" + "━" * 44)
    print("  STUDENT PERFORMANCE PREDICTION REPORT")
    print("━" * 44)
    print(f"  Study hours       : {inputs['study_hours']} hrs/day")
    print(f"  Attendance        : {inputs['attendance_pct']}%")
    print(f"  Previous score    : {inputs['prev_exam_score']}/100")
    print(f"  Assignments done  : {inputs['assignments_done']}/10")
    print(f"  Sleep hours       : {inputs['sleep_hours']} hrs/night")
    print("  " + "─" * 40)
    print(f"  Predicted Score   : {predicted:.1f} / 100")
    print(f"  Grade             : {grade}  —  {msg}")
    print("━" * 44 + "\n")


# =============================================================================
# MAIN — runs when you execute: python src/prediction.py directly
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Student Performance Prediction — src/prediction.py")
    print("=" * 55)

    # --- Step 1: Train and save the model ---
    print("\n[1] Training and saving the best model...")
    result = train_and_save_model()
    print(f"    Best model  : {result['best_model_name']}")
    print(f"    R2          : {result['R2']:.4f}")
    print(f"    RMSE        : {result['RMSE']:.4f}")
    print(f"    MAE         : {result['MAE']:.4f}")

    print("\n    All model results:")
    for name, metrics in result["all_results"].items():
        print(f"      {name:<25}  R2={metrics['R2']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}")

    # --- Step 2: Load model and make predictions ---
    print("\n[2] Loading saved model and making predictions...")
    saved_model = load_model()
    print(f"    Model loaded: {type(saved_model).__name__}")

    # --- Step 3: Example predictions ---
    print("\n[3] Example predictions:\n")

    examples = [
        ("High achiever",         9.0, 95.0, 88.0, 10, 7.5),
        ("Average student",       5.0, 74.0, 63.0,  5, 7.0),
        ("Struggling student",    1.5, 52.0, 38.0,  1, 5.0),
        ("Sleep-deprived grinder",8.0, 90.0, 80.0,  8, 4.5),
        ("Diligent but weak base",7.0, 88.0, 45.0,  9, 8.0),
    ]

    print(f"  {'Profile':<26} {'Study':>6} {'Attend':>7} {'Prev':>6} "
          f"{'Asgn':>5} {'Sleep':>6} {'Score':>8}")
    print("  " + "-" * 66)

    for label, sh, ap, ps, ad, slp in examples:
        score = predict_score(sh, ap, ps, ad, slp, model=saved_model)
        print(f"  {label:<26} {sh:>6.1f} {ap:>6.1f}% {ps:>6.1f} "
              f"{ad:>5} {slp:>6.1f}  {score:>7.1f}")

    # --- Step 4: Verbose example ---
    print("\n[4] Verbose prediction for one student:")
    predict_score(
        study_hours=6.5,
        attendance_pct=82.0,
        prev_exam_score=70.0,
        assignments_done=7,
        sleep_hours=7.0,
        verbose=True
    )

    # --- Step 5: Batch prediction ---
    print("[5] Batch prediction demo (5 students at once):")
    batch_df = pd.DataFrame({
        "study_hours":       [2.0, 5.5, 8.0, 3.0, 9.5],
        "attendance_pct":    [55.0, 72.0, 91.0, 60.0, 98.0],
        "prev_exam_score":   [40.0, 65.0, 82.0, 48.0, 90.0],
        "assignments_done":  [1,    5,    9,    2,    10],
        "sleep_hours":       [5.0,  7.0,  7.5,  6.0,  8.0],
    })
    batch_scores = predict_batch(batch_df, model=saved_model)
    batch_df["predicted_score"] = batch_scores.round(1)
    print(batch_df.to_string(index=False))

    print("\n  All done! src/prediction.py is fully functional.")
