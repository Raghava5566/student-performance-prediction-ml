# =============================================================================
# Stage 5: Best Model Selection + Save + Prediction Function
# File: notebooks/stage5_prediction.py
# Run: python notebooks/stage5_prediction.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle   # Built-in Python library for saving/loading Python objects to disk

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("notebooks/eda_plots", exist_ok=True)
os.makedirs("src", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

print("=" * 62)
print("  STAGE 5: Best Model Selection + Prediction Function")
print("=" * 62)

# =============================================================================
# SECTION 1: RETRAIN ALL MODELS (clean reproducible run)
# =============================================================================
print("\n[1] RETRAINING ALL MODELS")
print("-" * 40)

df = pd.read_csv("data/student_performance.csv")
FEATURES = ["study_hours", "attendance_pct", "prev_exam_score",
            "assignments_done", "sleep_hours"]
TARGET = "final_score"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train all 3 models
models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"    {name:<25}  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# =============================================================================
# SECTION 2: FORMAL BEST MODEL SELECTION
# =============================================================================
print("\n[2] FORMAL BEST MODEL SELECTION")
print("-" * 40)

# Find winner by R² (highest)
best_name  = max(results, key=lambda n: results[n]["R2"])
best_entry = results[best_name]
best_model = best_entry["model"]

print(f"""
  DECISION CRITERIA:
  We select the model with the HIGHEST R² on the test set.
  R² is preferred as the headline metric because it tells us
  the proportion of score variance the model explains —
  directly interpretable as "how well does the model understand
  what drives a student's final score?"

  Tiebreaker: If R² values were close (< 0.01 difference),
  we would prefer Linear Regression for its simplicity and
  interpretability (interviewers love this principle: Occam's Razor).

  WINNER: {best_name}
    R²   = {best_entry['R2']:.4f}  ({best_entry['R2']*100:.1f}% of variance explained)
    RMSE = {best_entry['RMSE']:.4f} marks (average prediction error)
    MAE  = {best_entry['MAE']:.4f} marks (average absolute error)

  WHY THIS MODEL WON (in plain language):
""")

if best_name == "Linear Regression":
    print("""    Linear Regression won because our dataset has a fundamentally
    linear structure — each feature contributes a fixed amount to
    the final score (study_hours × 3.5, etc.). The model matched
    this structure perfectly. Adding more complexity (Decision Tree,
    Random Forest) introduced unnecessary variance without improving
    accuracy. This demonstrates an important ML principle:
    "The best model is the simplest one that fits the data well."
""")
elif best_name == "Random Forest":
    print("""    Random Forest won because it combines 100 decision trees, each
    trained on slightly different data. The averaging cancels out
    individual tree overfitting, producing robust predictions.
    It also captures non-linear patterns (like the sleep-hours U-shape)
    that Linear Regression cannot represent as a straight line.
""")
else:
    print("""    Decision Tree won by building a custom decision path for each
    student's profile. Its flexible branching captured patterns that
    simpler linear models missed, without the overhead of an ensemble.
""")

# =============================================================================
# SECTION 3: SAVE THE BEST MODEL TO DISK
# =============================================================================
print("[3] SAVING THE BEST MODEL TO DISK")
print("-" * 40)

# WHY SAVE THE MODEL?
# ───────────────────────────────────────────────────────────────────────────
# Training takes time (even if just seconds here). In production, you train
# ONCE, save the trained model, then load it every time you need a prediction.
# This is called "model serialisation" — converting the Python object to bytes
# so it can be stored and re-loaded later.
#
# pickle is Python's standard way to do this.
# The saved file has extension .pkl (short for "pickle").
#
# Alternative: joblib (from scikit-learn) is faster for large numpy arrays.
# For this project, pickle is perfectly fine.

model_path = "src/best_model.pkl"
with open(model_path, "wb") as f:   # "wb" = write bytes (not text)
    pickle.dump(best_model, f)

model_size = os.path.getsize(model_path)
print(f"    Model saved to: {model_path}")
print(f"    File size: {model_size / 1024:.1f} KB")

# Verify: reload it and make a test prediction
with open(model_path, "rb") as f:   # "rb" = read bytes
    loaded_model = pickle.load(f)

# Quick sanity check — predictions should match exactly
test_pred_original = best_model.predict(X_test[:3])
test_pred_loaded   = loaded_model.predict(X_test[:3])
match = np.allclose(test_pred_original, test_pred_loaded)
print(f"    Reload verification: predictions match = {match}")

# Also save the feature list so prediction.py knows the column order
features_path = "src/feature_names.pkl"
with open(features_path, "wb") as f:
    pickle.dump(FEATURES, f)
print(f"    Feature names saved to: {features_path}")

# =============================================================================
# SECTION 4: BUILD THE PREDICTION FUNCTION
# =============================================================================
print("\n[4] DEMONSTRATING THE PREDICTION FUNCTION")
print("-" * 40)

# The actual reusable function lives in src/prediction.py (created separately).
# Here we define and demo it inline to show how it works step-by-step.

def predict_score(study_hours, attendance_pct, prev_exam_score,
                  assignments_done, sleep_hours,
                  model=best_model, verbose=True):
    """
    Predict a student's final exam score using the trained ML model.

    Parameters:
    -----------
    study_hours      : float  — Daily study hours (0.5 to 10)
    attendance_pct   : float  — Attendance percentage (50 to 100)
    prev_exam_score  : float  — Previous exam score (0 to 100)
    assignments_done : int    — Assignments completed (0 to 10)
    sleep_hours      : float  — Average nightly sleep (4 to 10)
    model            : trained sklearn model (default: best_model)
    verbose          : bool   — Print a detailed breakdown?

    Returns:
    --------
    float — Predicted final score (clipped to valid range 0-100)
    """

    # ── Step 1: Input Validation ──────────────────────────────────────────────
    # Before we let the model see the input, we check it makes sense.
    # Garbage in → garbage out. Validation catches impossible inputs early.
    errors = []
    if not (0.0 <= study_hours <= 24.0):
        errors.append(f"  study_hours={study_hours} is outside valid range [0, 24]")
    if not (0.0 <= attendance_pct <= 100.0):
        errors.append(f"  attendance_pct={attendance_pct} must be between 0 and 100")
    if not (0.0 <= prev_exam_score <= 100.0):
        errors.append(f"  prev_exam_score={prev_exam_score} must be between 0 and 100")
    if not (0 <= assignments_done <= 10):
        errors.append(f"  assignments_done={assignments_done} must be between 0 and 10")
    if not (0.0 <= sleep_hours <= 24.0):
        errors.append(f"  sleep_hours={sleep_hours} is outside valid range [0, 24]")
    if errors:
        raise ValueError("Invalid input(s):\n" + "\n".join(errors))

    # ── Step 2: Package Input as a DataFrame Row ──────────────────────────────
    # The model was trained on a DataFrame with named columns in a specific order.
    # We must provide input in exactly the same format — same column names, same order.
    # pd.DataFrame(...) with a list of one dict creates a single-row DataFrame.
    input_data = pd.DataFrame([{
        "study_hours":       study_hours,
        "attendance_pct":    attendance_pct,
        "prev_exam_score":   prev_exam_score,
        "assignments_done":  assignments_done,
        "sleep_hours":       sleep_hours,
    }])

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    # model.predict() returns an array even for one row → [predicted_value]
    # We take [0] to extract the single number from the array.
    raw_prediction = model.predict(input_data)[0]

    # ── Step 4: Clip to Valid Range ───────────────────────────────────────────
    # The model is a mathematical formula — it could theoretically output
    # a value slightly above 100 or below 0 for extreme inputs.
    # np.clip(value, min, max) ensures we never return an impossible score.
    final_prediction = float(np.clip(raw_prediction, 0, 100))

    # ── Step 5: Explain the Prediction (if verbose=True) ─────────────────────
    if verbose:
        print(f"\n  Student Profile:")
        print(f"    Study hours      : {study_hours} hrs/day")
        print(f"    Attendance       : {attendance_pct}%")
        print(f"    Previous score   : {prev_exam_score}/100")
        print(f"    Assignments done : {assignments_done}/10")
        print(f"    Sleep hours      : {sleep_hours} hrs/night")
        print(f"\n  Raw model output : {raw_prediction:.2f}")
        print(f"  Clipped output   : {final_prediction:.2f}")
        print(f"\n  ╔══════════════════════════════════╗")
        print(f"  ║  Predicted Final Score: {final_prediction:>5.1f}/100  ║")
        print(f"  ╚══════════════════════════════════╝")

        # Give a qualitative interpretation
        if final_prediction >= 90:
            grade, msg = "A", "Outstanding performance expected!"
        elif final_prediction >= 75:
            grade, msg = "B", "Good performance expected."
        elif final_prediction >= 60:
            grade, msg = "C", "Average performance expected."
        elif final_prediction >= 45:
            grade, msg = "D", "Below average — more effort needed."
        else:
            grade, msg = "F", "At risk — significant improvement needed."
        print(f"  Grade: {grade} — {msg}")

    return final_prediction


# =============================================================================
# SECTION 5: LIVE DEMO — 5 DIVERSE STUDENT PROFILES
# =============================================================================
print("\n[5] LIVE PREDICTION DEMO — 5 STUDENT PROFILES")
print("=" * 62)

test_students = [
    {
        "label": "Student A: The Star Performer",
        "study_hours": 9.0, "attendance_pct": 95.0,
        "prev_exam_score": 88.0, "assignments_done": 10, "sleep_hours": 7.5
    },
    {
        "label": "Student B: The Average Student",
        "study_hours": 5.0, "attendance_pct": 74.0,
        "prev_exam_score": 63.0, "assignments_done": 5, "sleep_hours": 7.0
    },
    {
        "label": "Student C: The Struggling Student",
        "study_hours": 1.5, "attendance_pct": 52.0,
        "prev_exam_score": 38.0, "assignments_done": 1, "sleep_hours": 5.0
    },
    {
        "label": "Student D: Smart but Sleep-Deprived",
        "study_hours": 8.0, "attendance_pct": 90.0,
        "prev_exam_score": 80.0, "assignments_done": 8, "sleep_hours": 4.5
    },
    {
        "label": "Student E: Good Habits, Weak Foundation",
        "study_hours": 7.0, "attendance_pct": 88.0,
        "prev_exam_score": 45.0, "assignments_done": 9, "sleep_hours": 8.0
    },
]

summary_rows = []
for student in test_students:
    label = student.pop("label")
    print(f"\n  -- {label} --")
    score = predict_score(**student, verbose=True)
    summary_rows.append({"Student": label, **student, "Predicted Score": round(score, 1)})

print("\n" + "=" * 62)
print("  PREDICTION SUMMARY TABLE")
print("=" * 62)
summary_df = pd.DataFrame(summary_rows).set_index("Student")
print(summary_df.to_string())

# =============================================================================
# SECTION 6: VISUALISE PREDICTIONS
# =============================================================================
print("\n[6] SAVING PREDICTION VISUALISATION")
print("-" * 40)

fig, ax = plt.subplots(figsize=(12, 5))

names   = [r["Student"].split(":")[1].strip() for r in summary_rows]
scores  = [r["Predicted Score"] for r in summary_rows]

# Grade-based colour coding
def score_color(s):
    if s >= 90: return "#2ECC71"   # green
    if s >= 75: return "#3498DB"   # blue
    if s >= 60: return "#F39C12"   # orange
    if s >= 45: return "#E67E22"   # dark orange
    return "#E74C3C"               # red

bar_cols = [score_color(s) for s in scores]
bars = ax.barh(names, scores, color=bar_cols, edgecolor="white",
               linewidth=1.5, height=0.55)

# Add score labels
for bar, score in zip(bars, scores):
    ax.text(score + 0.5, bar.get_y() + bar.get_height()/2,
            f"{score:.1f}/100", va="center", fontweight="bold", fontsize=11)

# Grade zone shading
for lo, hi, color, label in [(90, 100, "#2ECC71", "A"), (75, 90, "#3498DB", "B"),
                               (60, 75, "#F39C12", "C"), (45, 60, "#E67E22", "D"),
                               (0,  45, "#E74C3C", "F")]:
    ax.axvspan(lo, hi, alpha=0.06, color=color)
    ax.text((lo+hi)/2, -0.65, label, ha="center", fontsize=10,
            color=color, fontweight="bold")

ax.set_xlabel("Predicted Final Score (out of 100)")
ax.set_title("Predicted Scores for 5 Diverse Student Profiles",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 108)
ax.axvline(60, color="gray", linestyle=":", lw=1.2, label="Pass mark (60)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("notebooks/eda_plots/12_predictions_demo.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/12_predictions_demo.png")

print("\n" + "=" * 62)
print("  STAGE 5 COMPLETE")
print("  Best model saved to: src/best_model.pkl")
print("  Prediction function ready in: src/prediction.py (next step)")
print("=" * 62)
