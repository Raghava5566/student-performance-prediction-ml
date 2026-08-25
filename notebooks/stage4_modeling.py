# =============================================================================
# Stage 4: Model Training, Evaluation & Comparison
# File: notebooks/stage4_modeling.py
# Run: python notebooks/stage4_modeling.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-learn is the core ML library in Python.
# It provides ready-made implementations of dozens of ML algorithms.
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("notebooks/eda_plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

print("=" * 62)
print("  STAGE 4: Model Training, Evaluation & Comparison")
print("=" * 62)

# =============================================================================
# SECTION 1: LOAD DATA AND DEFINE X, y
# =============================================================================
print("\n[1] LOADING DATA AND DEFINING X, y")
print("-" * 40)

df = pd.read_csv("data/student_performance.csv")

FEATURES = ["study_hours", "attendance_pct", "prev_exam_score",
            "assignments_done", "sleep_hours"]
TARGET   = "final_score"

X = df[FEATURES]  # feature matrix (500 x 5)
y = df[TARGET]    # target vector (500,)

print(f"    X shape: {X.shape}  (students x features)")
print(f"    y shape: {y.shape}  (target scores)")

# =============================================================================
# SECTION 2: TRAIN / TEST SPLIT
# =============================================================================
print("\n[2] TRAIN / TEST SPLIT")
print("-" * 40)

# WHY split the data?
# ─────────────────────────────────────────────────────────────────────────────
# Imagine studying for an exam using old question papers.
# You memorise all the answers. Then the SAME questions appear in the real exam.
# You'd score 100% — but you didn't actually LEARN, you just memorised.
#
# That's what happens if you train AND test on the same data.
# The model would seem perfect but fail on new students.
#
# SOLUTION: Keep some data hidden from the model during training (the "test set").
# Train on 80% of the data → evaluate on the remaining 20%.
# The model has NEVER seen the test set → gives an honest performance estimate.

X_train, X_test, y_train, y_test = train_test_split(
    X,           # features
    y,           # target
    test_size=0.2,    # 20% goes to testing (100 students)
    random_state=42   # seed for reproducibility — same split every run
)

print(f"""
    Total data     : {len(X)} students
    Training set   : {len(X_train)} students (80%) -- model LEARNS from these
    Test set       : {len(X_test)} students (20%) -- model is EVALUATED on these

    KEY INSIGHT:
    The test set is like a "sealed envelope" that the model never opens
    during training. Only after training is complete do we open it to
    measure how well the model generalised to new, unseen data.
""")

# =============================================================================
# SECTION 3: TRAIN ALL THREE MODELS
# =============================================================================
print("[3] TRAINING THREE MODELS")
print("-" * 40)

# ─── MODEL 1: LINEAR REGRESSION ───────────────────────────────────────────────
# WHAT IT DOES:
#   Fits a straight line (or flat plane in 5D) through the data.
#   The model learns one coefficient (weight) for each feature:
#
#   final_score = w1*study_hours + w2*attendance_pct + w3*prev_score
#               + w4*assignments + w5*sleep_hours + bias
#
#   It finds the w values that minimise prediction error across all 400 students.
#
# ANALOGY: Drawing the single best-fit straight line through a scatter plot.
#
# STRENGTH : Simple, fast, easy to interpret. Ideal if the real relationship
#            between features and target IS roughly linear.
# WEAKNESS : Cannot capture non-linear patterns (e.g., sleep_hours U-shape).

print("\n  Training Model 1: Linear Regression...")
lr_model = LinearRegression()
# .fit() = the training step. The model reads X_train and y_train,
# figures out the best line/plane, and stores the coefficients internally.
lr_model.fit(X_train, y_train)
# .predict() = use the learned line to generate predictions for new data
lr_preds = lr_model.predict(X_test)
print("    Done. Coefficients learned:")
for feat, coef in zip(FEATURES, lr_model.coef_):
    print(f"      {feat:<22}: {coef:+.4f}  "
          f"(each unit increase raises score by {coef:.2f} points)")
print(f"      bias (intercept)      : {lr_model.intercept_:.4f}")

# ─── MODEL 2: DECISION TREE REGRESSOR ─────────────────────────────────────────
# WHAT IT DOES:
#   Builds a tree of yes/no questions about the features.
#   Example path through the tree:
#     "Is study_hours > 6?"
#       YES → "Is attendance_pct > 80?"
#               YES → predict 88.5
#               NO  → predict 76.2
#       NO  → "Is prev_exam_score > 70?"
#               YES → predict 68.4
#               NO  → predict 52.1
#
#   The tree keeps splitting until it reaches a "leaf" node — a final prediction.
#
# ANALOGY: A flowchart that asks questions and narrows down to an answer.
#
# STRENGTH : Captures non-linear patterns, no assumptions about data shape.
#            Easy to visualise. Can overfit easily.
# WEAKNESS : Without limits (max_depth), it memorises training data perfectly
#            (overfitting) but generalises poorly to new data.
#            max_depth=10 limits the tree to 10 levels of questions.

print("\n  Training Model 2: Decision Tree Regressor...")
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
# max_depth=10: don't let the tree grow deeper than 10 splits
# This prevents overfitting — a shallow tree generalises better
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
print(f"    Done. Tree depth: {dt_model.get_depth()} levels")
print(f"    Number of leaf nodes: {dt_model.get_n_leaves()}")

# ─── MODEL 3: RANDOM FOREST REGRESSOR ─────────────────────────────────────────
# WHAT IT DOES:
#   Builds MANY decision trees (100 by default) — each on a random subset of:
#     - the training data (random rows)
#     - the features (random columns)
#   Each tree gives its own prediction.
#   The final prediction is the AVERAGE of all 100 trees' predictions.
#
# ANALOGY: Instead of asking one expert, ask 100 different experts and average
#          their opinions. One expert might be wrong; the crowd is usually right.
#
# WHY IS THIS BETTER?
#   Each tree is slightly different (random data/features).
#   Some trees overfit in one direction, others in another direction.
#   When you average them, the overfitting cancels out!
#   This is called "ensemble learning" — combining many weak models into one strong one.
#
# STRENGTH : Very accurate, handles non-linearity, resistant to overfitting.
# WEAKNESS : Slower to train, harder to interpret than a single tree.

print("\n  Training Model 3: Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# n_estimators=100: build 100 decision trees
# max_depth=10: each tree has at most 10 levels of splits
# random_state=42: reproducible results
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(f"    Done. Forest contains: {rf_model.n_estimators} trees")

# =============================================================================
# SECTION 4: EVALUATE ALL THREE MODELS
# =============================================================================
print("\n[4] EVALUATING MODELS WITH 4 METRICS")
print("-" * 40)

# METRICS EXPLAINED:
# ─────────────────────────────────────────────────────────────────────────────
# We need to quantify "how wrong" each model is. Here are 4 standard metrics:
#
# 1. MAE (Mean Absolute Error)
#    = average of |actual - predicted| for all test students
#    Intuition: "On average, the model's prediction is off by MAE points."
#    Example: MAE = 5.2 means predictions are wrong by ~5.2 marks on average.
#    Lower is better. Easy to understand.
#
# 2. MSE (Mean Squared Error)
#    = average of (actual - predicted)² for all test students
#    WHY SQUARE? Squaring makes big errors count WAY more than small errors.
#    A 10-mark error counts 4x more than a 5-mark error (100 vs 25).
#    This punishes large mistakes more severely than MAE does.
#    Lower is better. Harder to interpret (unit is marks²).
#
# 3. RMSE (Root Mean Squared Error)
#    = square root of MSE
#    Brings MSE back to the same unit as the target (marks, not marks²).
#    Intuition: "Typical error size, with big errors penalised more."
#    Lower is better. Most commonly reported metric.
#
# 4. R² (R-squared, "coefficient of determination")
#    = how much of the variation in scores did the model explain?
#    Range: 0 to 1 (sometimes negative for very bad models)
#    R² = 1.0 → perfect predictions
#    R² = 0.0 → model is as good as just predicting the mean every time
#    R² = 0.85 → model explains 85% of the variation in scores
#    Higher is better.

def evaluate(name, y_true, y_pred):
    """Calculate and return all 4 metrics for a model."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

results = [
    evaluate("Linear Regression",     y_test, lr_preds),
    evaluate("Decision Tree",          y_test, dt_preds),
    evaluate("Random Forest",          y_test, rf_preds),
]
results_df = pd.DataFrame(results).set_index("Model")

print("\n  COMPARISON TABLE:")
print(results_df.round(4).to_string())

# Identify the winner
best_model_name = results_df["R²"].idxmax()
best_r2         = results_df["R²"].max()
best_rmse       = results_df.loc[best_model_name, "RMSE"]
print(f"\n  WINNER: {best_model_name}")
print(f"    R²   = {best_r2:.4f}  (explains {best_r2*100:.1f}% of score variance)")
print(f"    RMSE = {best_rmse:.4f} marks (typical prediction error)")

# =============================================================================
# SECTION 5: VISUALISE RESULTS
# =============================================================================
print("\n[5] SAVING RESULT VISUALISATIONS...")
print("-" * 40)

model_names = ["Linear\nRegression", "Decision\nTree", "Random\nForest"]
bar_colors  = ["#4C72B0", "#DD8452", "#55A868"]

# ── PLOT 8: Metric Comparison Bar Charts ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Model Comparison: All 4 Evaluation Metrics",
             fontsize=15, fontweight="bold", y=1.02)

metrics = ["MAE", "MSE", "RMSE", "R²"]
for i, (metric, ax) in enumerate(zip(metrics, axes)):
    vals = [results_df.loc[m.replace("\n", " "), metric]
            for m in model_names]
    # For MAE/MSE/RMSE: lower = better, highlight minimum
    # For R²: higher = better, highlight maximum
    best_idx = vals.index(min(vals)) if metric != "R²" else vals.index(max(vals))

    bars = ax.bar(model_names, vals, color=bar_colors,
                  edgecolor="white", linewidth=1.5, width=0.5)

    # Put a gold star on the winning bar
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    # Add value labels on top of each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title(f"{metric}\n({'lower=better' if metric != 'R²' else 'higher=better'})",
                 fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("notebooks/eda_plots/08_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/08_model_comparison.png")

# ── PLOT 9: Actual vs Predicted Scatter (all 3 models) ────────────────────────
# The perfect model would lie exactly on a diagonal line y=x.
# Scatter around that line = prediction error.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Actual vs Predicted Final Scores (Test Set)",
             fontsize=14, fontweight="bold", y=1.02)

model_data = [
    ("Linear Regression", lr_preds, "#4C72B0"),
    ("Decision Tree",      dt_preds, "#DD8452"),
    ("Random Forest",      rf_preds, "#55A868"),
]

for ax, (name, preds, color) in zip(axes, model_data):
    ax.scatter(y_test, preds, alpha=0.45, color=color,
               edgecolors="none", s=22)

    # The "perfect prediction" diagonal line — if a model were perfect,
    # all dots would lie exactly on this line.
    lo = min(y_test.min(), preds.min())
    hi = max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")

    r2  = results_df.loc[name, "R²"]
    rmse= results_df.loc[name, "RMSE"]
    ax.set_title(f"{name}\nR²={r2:.3f}  RMSE={rmse:.2f}",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Actual Final Score")
    ax.set_ylabel("Predicted Final Score")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("notebooks/eda_plots/09_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/09_actual_vs_predicted.png")

# ── PLOT 10: Feature Importance (Random Forest) ────────────────────────────────
# Random Forest can tell us: "Which features were most useful for prediction?"
# It measures how much each feature reduced prediction error across all 100 trees.
# Higher importance → that feature was used more and helped more.
importances = rf_model.feature_importances_  # array of 5 values summing to 1.0
feat_imp_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
               color=bar_colors[:len(FEATURES)], edgecolor="white", height=0.55)

# Label each bar with its importance value
for bar, val in zip(bars, feat_imp_df["Importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f} ({val*100:.1f}%)",
            va="center", fontsize=10, fontweight="bold")

ax.set_xlabel("Feature Importance (fraction of total contribution)")
ax.set_title("Random Forest Feature Importance\n"
             "(Which features matter most for prediction?)",
             fontweight="bold", fontsize=13)
ax.set_xlim(0, feat_imp_df["Importance"].max() * 1.3)
plt.tight_layout()
plt.savefig("notebooks/eda_plots/10_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/10_feature_importance.png")

# ── PLOT 11: Residuals (Prediction Errors) for Best Model ─────────────────────
# A residual = actual score - predicted score
# Positive residual → model under-predicted (student did better than expected)
# Negative residual → model over-predicted (student did worse than expected)
# IDEAL: residuals should be randomly scattered around 0 (no systematic bias)
# If residuals show a pattern, the model is making systematic mistakes.

best_preds  = rf_preds
residuals   = y_test.values - best_preds

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Residual Analysis — Random Forest (Best Model)",
             fontsize=14, fontweight="bold")

# Left: residuals vs predicted value
axes[0].scatter(best_preds, residuals, alpha=0.45,
                color="#55A868", edgecolors="none", s=22)
axes[0].axhline(0, color="red", linestyle="--", lw=2,
                label="Zero error line")
axes[0].set_xlabel("Predicted Final Score")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].set_title("Residuals vs Predicted Values\n"
                  "(Random scatter = good; patterns = bad)")
axes[0].legend()

# Right: histogram of residuals — should be bell-shaped around 0
axes[1].hist(residuals, bins=25, color="#55A868", edgecolor="white",
             linewidth=0.6)
axes[1].axvline(0, color="red", linestyle="--", lw=2)
axes[1].axvline(residuals.mean(), color="orange", linestyle="--", lw=2,
                label=f"Mean residual: {residuals.mean():.2f}")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Residuals\n"
                  "(Bell-shaped around 0 = ideal)")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebooks/eda_plots/11_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/11_residuals.png")

# =============================================================================
# SECTION 6: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 62)
print("  STAGE 4 COMPLETE — RESULTS SUMMARY")
print("=" * 62)

print("\n  COMPARISON TABLE (rounded to 4 decimal places):")
print()
header = f"  {'Model':<25} {'MAE':>8} {'MSE':>10} {'RMSE':>8} {'R²':>8}"
print(header)
print("  " + "-" * (len(header) - 2))
for name, row in results_df.iterrows():
    star = " <-- WINNER" if name == best_model_name else ""
    print(f"  {name:<25} {row['MAE']:>8.4f} {row['MSE']:>10.4f} "
          f"{row['RMSE']:>8.4f} {row['R²']:>8.4f}{star}")

print(f"""
  WHY RANDOM FOREST WON:
  - Higher R² = explains more of the variation in student scores
  - Lower RMSE = smaller average prediction error
  - Robust to overfitting (100 trees average out individual mistakes)
  - Captures non-linear patterns (e.g., sleep hours U-shape)
    that Linear Regression misses entirely

  WHAT R² = {best_r2:.3f} MEANS:
  The Random Forest model explains {best_r2*100:.1f}% of the variation
  in final scores. The remaining {(1-best_r2)*100:.1f}% is due to factors
  we didn't capture (mood, test anxiety, teaching quality, etc.)

  --> Ready for Stage 5: Prediction Function!
""")
print("=" * 62)
