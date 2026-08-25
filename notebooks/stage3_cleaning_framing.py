# =============================================================================
# Stage 3: Data Cleaning + Problem Framing
# File: notebooks/stage3_cleaning_framing.py
# Run: python notebooks/stage3_cleaning_framing.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

os.makedirs("notebooks/eda_plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

print("=" * 62)
print("  STAGE 3: Data Cleaning + Problem Framing")
print("=" * 62)

# Load our original clean dataset
df_clean = pd.read_csv("data/student_performance.csv")

# =============================================================================
# SECTION 1: SIMULATE A "DIRTY" DATASET
# =============================================================================
# Our dataset happens to be clean. But in the real world, data is ALWAYS messy.
# To teach you the cleaning techniques, we'll create a messy copy of the data,
# clean it step by step, and prove it matches the original.

print("\n[1] CREATING A SIMULATED DIRTY DATASET")
print("-" * 40)
print("    (Injecting problems into a copy so we can learn to fix them)")

np.random.seed(99)  # reproducibility

# Copy the clean data — we'll mess this up
df = df_clean.copy()

N = len(df)  # 500 rows

# --- Problem 1: Inject 30 missing values spread across 4 columns ---
# In real datasets, missing values occur when:
#   - A student didn't submit attendance records
#   - A sensor failed to record sleep data
#   - A form field was left blank
missing_indices = np.random.choice(N, 30, replace=False)
df.loc[missing_indices[:10], "study_hours"]      = np.nan   # NaN = "Not a Number" (missing)
df.loc[missing_indices[10:20], "attendance_pct"] = np.nan
df.loc[missing_indices[20:25], "sleep_hours"]    = np.nan
df.loc[missing_indices[25:], "assignments_done"] = np.nan

# --- Problem 2: Inject 12 exact duplicate rows ---
# Duplicates happen from double-form submissions, data pipeline errors, etc.
dup_rows = df.iloc[5:17].copy()   # take 12 rows from the middle
df = pd.concat([df, dup_rows], ignore_index=True)  # append them to the end

# --- Problem 3: Inject 3 outliers ---
# Outliers = values that are physically impossible or extremely unusual
df.loc[500, "study_hours"]   = 99.0   # nobody studies 99 hours a day!
df.loc[501, "sleep_hours"]   = 0.0    # sleeping 0 hours is impossible
df.loc[502, "attendance_pct"] = 150.0  # attendance can't exceed 100%

print(f"    Dirty dataset shape: {df.shape}")
print(f"    (Original was 500 rows. Added 12 duplicates + 3 outlier rows = 515 total)")

# =============================================================================
# SECTION 2: STEP-BY-STEP CLEANING
# =============================================================================

print("\n[2] STEP-BY-STEP DATA CLEANING")
print("-" * 40)

# ── STEP 2a: Inspect Missing Values ──────────────────────────────────────────
print("\n  STEP A: Check for missing values")
missing = df.isnull().sum()
print(f"\n  Missing values per column:")
for col, count in missing.items():
    bar = "!" * count if count > 0 else "OK"
    print(f"    {col:<22} {count:>3} missing  [{bar}]")

total_missing = missing.sum()
pct_missing   = (total_missing / (df.shape[0] * df.shape[1])) * 100
print(f"\n  Total missing: {total_missing} cells ({pct_missing:.1f}% of all data)")

# ── STEP 2b: Handle Missing Values ───────────────────────────────────────────
print("\n  STEP B: Handle missing values")
print("""
  Three main strategies for missing data:

  Strategy 1 — DROP rows:
    df.dropna()
    USE WHEN: Very few rows are missing, and you have plenty of data.
    RISK: You lose data. If 30% is missing, you lose 30% of your dataset!

  Strategy 2 — IMPUTE with mean/median:
    df['col'].fillna(df['col'].mean())
    USE WHEN: The column is numerical, and missing values seem random.
    WHY MEDIAN not always MEAN? Mean is pulled by outliers. Median isn't.
    e.g., if 4 students score [80, 85, 82, 0] (0 = missing):
      mean = 61.75 (pulled down by 0)
      median = 82.5 (robust — not affected by extreme values)

  Strategy 3 — IMPUTE with mode:
    df['col'].fillna(df['col'].mode()[0])
    USE WHEN: The column is categorical (text) or discrete (whole numbers)
    e.g., fill missing 'assignments_done' with the most common value
  """)

# APPLY: Use median for continuous columns, mode for discrete
for col in ["study_hours", "attendance_pct", "sleep_hours"]:
    before_count = df[col].isnull().sum()
    if before_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"    '{col}': filled {before_count} NaNs with median ({median_val:.2f})")

# assignments_done is a whole number → use mode
col = "assignments_done"
before_count = df[col].isnull().sum()
if before_count > 0:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"    '{col}': filled {before_count} NaNs with mode ({mode_val})")

print(f"\n    Missing values after cleaning: {df.isnull().sum().sum()}")

# ── STEP 2c: Remove Duplicate Rows ───────────────────────────────────────────
print("\n  STEP C: Remove duplicate rows")
dup_count = df.duplicated().sum()
print(f"    Duplicates found: {dup_count}")
df = df.drop_duplicates()
# reset_index: after dropping rows, index numbers have gaps (0,1,5,6...)
# reset_index() re-numbers them cleanly (0,1,2,3...)
df = df.reset_index(drop=True)
# drop=True means: don't save the old index as a new column
print(f"    Duplicates removed. New shape: {df.shape}")

# ── STEP 2d: Handle Outliers ─────────────────────────────────────────────────
print("\n  STEP D: Detect and handle outliers")
print("""
  Two main outlier detection methods:

  Method 1 — IQR (Interquartile Range) — the standard approach:
    IQR  = Q3 - Q1   (the spread of the middle 50% of data)
    Lower fence = Q1 - 1.5 * IQR   (anything below = outlier)
    Upper fence = Q3 + 1.5 * IQR   (anything above = outlier)
    This is exactly what the box plot whiskers show!

  Method 2 — Domain Knowledge (common sense):
    study_hours > 24 → impossible (more than hours in a day)
    attendance_pct > 100 → impossible
    sleep_hours < 0 → impossible
  """)

# Apply domain-knowledge clipping for physically impossible values
domain_rules = {
    "study_hours":      (0.0, 20.0),   # max 20 hrs/day is extreme but possible
    "attendance_pct":   (0.0, 100.0),  # can't exceed 100%
    "sleep_hours":      (1.0, 14.0),   # min 1hr, max 14hr
    "assignments_done": (0,   10),     # 0 to 10 assignments
    "prev_exam_score":  (0.0, 100.0),
    "final_score":      (0.0, 100.0),
}

for col, (lo, hi) in domain_rules.items():
    outliers = ((df[col] < lo) | (df[col] > hi)).sum()
    if outliers > 0:
        print(f"    '{col}': {outliers} outliers clipped to [{lo}, {hi}]")
        df[col] = df[col].clip(lower=lo, upper=hi)
    else:
        print(f"    '{col}': No domain-knowledge outliers found")

print(f"\n    Final cleaned shape: {df.shape}")
print(f"    Comparison: Original clean = (500, 6), After cleaning = {df.shape}")

# =============================================================================
# SECTION 3: VISUALISE "BEFORE vs AFTER" CLEANING
# =============================================================================
print("\n[3] SAVING BEFORE vs AFTER CLEANING VISUALISATION")
print("-" * 40)

# We'll plot the dirty vs clean distributions for study_hours (most affected)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Before vs After Cleaning — study_hours Distribution",
             fontsize=14, fontweight="bold")

# Before (dirty)
dirty_study = df_clean["study_hours"].copy()
# Re-inject the outlier for illustration
dirty_study_plot = pd.concat([dirty_study, pd.Series([99.0])], ignore_index=True)
sns.histplot(dirty_study_plot, kde=True, ax=axes[0], color="#C44E52",
             edgecolor="white", linewidth=0.5)
axes[0].set_title("BEFORE Cleaning (with outlier at 99)", fontweight="bold", color="#C44E52")
axes[0].set_xlabel("study_hours")
axes[0].set_ylabel("Number of Students")
axes[0].axvline(dirty_study_plot.mean(), color="black", linestyle="--",
                label=f"Mean: {dirty_study_plot.mean():.2f}")
axes[0].legend()

# After (clean)
sns.histplot(df["study_hours"], kde=True, ax=axes[1], color="#55A868",
             edgecolor="white", linewidth=0.5)
axes[1].set_title("AFTER Cleaning (outlier removed)", fontweight="bold", color="#55A868")
axes[1].set_xlabel("study_hours")
axes[1].set_ylabel("Number of Students")
axes[1].axvline(df["study_hours"].mean(), color="black", linestyle="--",
                label=f"Mean: {df['study_hours'].mean():.2f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebooks/eda_plots/06_before_after_cleaning.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/06_before_after_cleaning.png")

# =============================================================================
# SECTION 4: PROBLEM FRAMING — WHY REGRESSION?
# =============================================================================
print("\n[4] PROBLEM FRAMING: WHY IS THIS REGRESSION, NOT CLASSIFICATION?")
print("-" * 40)

print("""
  ┌─────────────────────────────────────────────────────────────┐
  │  THE GOLDEN RULE:                                           │
  │                                                             │
  │  Look at your TARGET variable.                              │
  │  Ask: "Is it a NUMBER or a CATEGORY?"                       │
  │                                                             │
  │  NUMBER (continuous) → REGRESSION                           │
  │  CATEGORY (label)   → CLASSIFICATION                        │
  └─────────────────────────────────────────────────────────────┘
  """)

print("  OUR CASE:")
print(f"  Target variable: final_score")
print(f"  Sample values  : {list(df['final_score'].head(8).round(1).values)}")
print(f"  Min: {df['final_score'].min()}  Max: {df['final_score'].max()}")
print("""
  These are CONTINUOUS NUMBERS on a scale from 0 to 100.
  Predicting a number → this is a REGRESSION problem.
  """)

print("  CONTRAST: When would it be CLASSIFICATION?")
print("""
  If instead of predicting the score, we wanted to predict:
    - "Will this student PASS or FAIL?"     → 2 classes (Binary Classification)
    - "Grade: A / B / C / D / F?"           → 5 classes (Multi-class Classification)
    - "Is this student at-risk: Yes or No?" → 2 classes (Binary Classification)

  In those cases, the output would be a LABEL (a word/category),
  not a number. That's the core difference.
  """)

print("  ANALOGY TO REMEMBER:")
print("""
  Regression  = asking "HOW MUCH?" or "HOW MANY?"
                e.g., "How many marks will this student get?"

  Classification = asking "WHICH ONE?" or "WHAT TYPE?"
                   e.g., "Will this student pass or fail?"
  """)

# =============================================================================
# SECTION 5: VISUALISE REGRESSION vs CLASSIFICATION
# =============================================================================
print("[5] SAVING REGRESSION vs CLASSIFICATION VISUALISATION")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Regression vs Classification — Understanding the Difference",
             fontsize=14, fontweight="bold", y=1.02)

# --- Left plot: Regression (our task) ---
ax = axes[0]
ax.set_title("REGRESSION\n(Predicting a continuous score)",
             fontweight="bold", fontsize=12, color="#4C72B0")

# Scatter points coloured by score
scatter = ax.scatter(
    df["study_hours"], df["final_score"],
    c=df["final_score"], cmap="viridis", alpha=0.5, s=15, edgecolors="none"
)
# Trend line
z = np.polyfit(df["study_hours"], df["final_score"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
ax.plot(x_line, p(x_line), "r--", lw=2.5, label="Predicted score (line)")

ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Score (continuous: 0–100)")
ax.legend()
plt.colorbar(scatter, ax=ax, label="Final Score")
ax.text(0.05, 0.95, "Output: a NUMBER\n(e.g., 73.4, 88.1, 62.5)",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#D6EAF8", alpha=0.8),
        verticalalignment="top")

# --- Right plot: Classification (what it would look like) ---
ax = axes[1]
ax.set_title("CLASSIFICATION\n(Predicting a grade label)",
             fontweight="bold", fontsize=12, color="#C44E52")

# Convert scores to grade labels just for illustration
grade_map = {"A (90-100)": (90, 100), "B (75-89)": (75, 90),
             "C (60-74)": (60, 75), "D (<60)": (0, 60)}
colors_cls = {"A (90-100)": "#2ECC71", "B (75-89)": "#3498DB",
              "C (60-74)": "#F39C12", "D (<60)": "#E74C3C"}

for grade, (lo, hi) in grade_map.items():
    mask = (df["final_score"] >= lo) & (df["final_score"] < hi)
    ax.scatter(
        df.loc[mask, "study_hours"],
        df.loc[mask, "final_score"],
        color=colors_cls[grade], alpha=0.6, s=15,
        label=grade, edgecolors="none"
    )

# Horizontal band shading for grade zones
for grade, (lo, hi) in grade_map.items():
    ax.axhspan(lo, hi, alpha=0.06, color=colors_cls[grade])
ax.axhline(90, color="#2ECC71", lw=0.8, linestyle=":")
ax.axhline(75, color="#3498DB", lw=0.8, linestyle=":")
ax.axhline(60, color="#F39C12", lw=0.8, linestyle=":")

ax.set_xlabel("Study Hours")
ax.set_ylabel("Final Score (used to assign grade label)")
ax.legend(title="Grade", fontsize=8, title_fontsize=9)
ax.text(0.05, 0.95, "Output: a LABEL\n(e.g., 'A', 'B', 'Pass', 'Fail')",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FADBD8", alpha=0.8),
        verticalalignment="top")

plt.tight_layout()
plt.savefig("notebooks/eda_plots/07_regression_vs_classification.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: notebooks/eda_plots/07_regression_vs_classification.png")

# =============================================================================
# SECTION 6: PREPARE X AND y FOR MODELLING
# =============================================================================
print("\n[6] PREPARING FEATURES (X) AND TARGET (y) FOR MODELLING")
print("-" * 40)

# We now use the CLEANED dataset (df) — not the original copy
# We drop 'final_score' from X because that's what we're trying to PREDICT.
# If we left it in X, the model would just memorize: "final_score predicts final_score" — useless!
FEATURES = ["study_hours", "attendance_pct", "prev_exam_score",
            "assignments_done", "sleep_hours"]
TARGET   = "final_score"

X = df[FEATURES]   # DataFrame with 5 columns — the inputs to the model
y = df[TARGET]     # Series with 1 column — what the model must predict

print(f"\n  X (features) shape: {X.shape}  ← 500 students, 5 features each")
print(f"  y (target)   shape: {y.shape}  ← 500 scores to predict")
print(f"\n  X.head(3):")
print(X.head(3).to_string(index=True))
print(f"\n  y.head(3):")
print(y.head(3).to_string())

# Save the cleaned dataset
df.to_csv("data/student_performance_cleaned.csv", index=False)
print(f"\n  Cleaned dataset saved: data/student_performance_cleaned.csv")
print(f"  Shape: {df.shape}")

# =============================================================================
# SECTION 7: STAGE 3 SUMMARY
# =============================================================================
print("\n" + "=" * 62)
print("  STAGE 3 SUMMARY")
print("=" * 62)
print("""
  DATA CLEANING STEPS COVERED:
  1. Detect missing values (df.isnull().sum())
  2. Handle missing values:
       - Continuous cols: fill with MEDIAN (robust to outliers)
       - Discrete cols  : fill with MODE (most common value)
  3. Remove duplicates (df.drop_duplicates())
  4. Handle outliers:
       - Domain-knowledge clipping (impossible values)
       - IQR method (statistical method)

  PROBLEM FRAMING:
  - Target (final_score) is a continuous number (0-100)
  - Predicting a continuous number = REGRESSION problem
  - If target were a category (Pass/Fail/Grade) = Classification

  X AND y DEFINED:
  - X: 5 feature columns (what we know about a student)
  - y: final_score column (what we want to predict)

  --> Ready for Stage 4: Model Training!
""")
print("=" * 62)
