# =============================================================================
# Stage 2: Exploratory Data Analysis (EDA)
# File: notebooks/eda_stage2.py
# Run this with: python notebooks/eda_stage2.py
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# Every library below serves a specific purpose:

import pandas as pd          # For loading and manipulating the dataset (DataFrames)
import numpy as np           # For numerical operations
import matplotlib            # The core plotting engine
matplotlib.use('Agg')        # Use non-interactive backend (saves files instead of popups)
import matplotlib.pyplot as plt  # pyplot is the "easy interface" to matplotlib
import seaborn as sns        # Seaborn builds on matplotlib — prettier statistical charts
import os

# ── OUTPUT FOLDER ─────────────────────────────────────────────────────────────
# We'll save all plots here so you can view them after the script runs
os.makedirs("notebooks/eda_plots", exist_ok=True)

# Set a visual style for all seaborn plots — "whitegrid" gives clean white
# background with subtle grid lines. Makes charts look professional.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

print("=" * 62)
print("  STAGE 2: Exploratory Data Analysis (EDA)")
print("  Student Performance Prediction Project")
print("=" * 62)

# =============================================================================
# SECTION 1: LOADING THE DATA
# =============================================================================
print("\n[1] LOADING THE DATASET")
print("-" * 40)

# pd.read_csv() reads a CSV file from disk and converts it into a DataFrame.
# A DataFrame is like an Excel spreadsheet living inside Python —
# rows are students, columns are their attributes.
df = pd.read_csv("data/student_performance.csv")

print(f"    File loaded successfully!")
print(f"    Path: data/student_performance.csv")

# =============================================================================
# SECTION 2: BASIC OVERVIEW
# =============================================================================
print("\n[2] BASIC OVERVIEW")
print("-" * 40)

# df.shape returns a tuple: (number_of_rows, number_of_columns)
# Think of it as: (how many students, how many attributes per student)
print(f"\n  df.shape --> {df.shape}")
print(f"  Rows    : {df.shape[0]}  (one row = one student)")
print(f"  Columns : {df.shape[1]}  (one column = one attribute)")

# df.head(n) shows the first n rows — your "first look" at the data
print("\n  df.head(5) -- First 5 rows:")
print(df.head(5).to_string(index=True))

# =============================================================================
# SECTION 3: df.info() — COLUMN TYPES AND NULLS
# =============================================================================
print("\n[3] COLUMN DATA TYPES (df.info())")
print("-" * 40)
# df.info() is your quick health check:
#   - Shows each column name
#   - Shows how many NON-NULL (non-missing) values it has
#   - Shows the data type: int64 (whole number), float64 (decimal), object (text)
# If any column shows fewer than 500 non-null values, we have MISSING DATA!
df.info()

# =============================================================================
# SECTION 4: df.describe() — STATISTICAL SUMMARY
# =============================================================================
print("\n[4] STATISTICAL SUMMARY (df.describe())")
print("-" * 40)
# df.describe() gives 8 key statistics for every numeric column:
#   count  = how many non-missing values
#   mean   = average value
#   std    = standard deviation (how spread out values are)
#   min    = smallest value
#   25%    = 25th percentile (25% of values are below this)
#   50%    = median (the middle value)
#   75%    = 75th percentile (75% of values are below this)
#   max    = largest value
print(df.describe().round(2).to_string())

# =============================================================================
# SECTION 5: MISSING VALUE CHECK
# =============================================================================
print("\n[5] MISSING VALUES CHECK")
print("-" * 40)
# df.isnull() creates a True/False table: True where a value is missing
# .sum() counts the Trues per column
# A good dataset should have 0 missing values in all columns!
missing = df.isnull().sum()
print("\n  Missing values per column:")
print(missing.to_string())

total_missing = missing.sum()
if total_missing == 0:
    print("\n  RESULT: No missing values found! Dataset is clean.")
else:
    print(f"\n  RESULT: {total_missing} missing values found! (Would need cleaning)")

# =============================================================================
# SECTION 6: DUPLICATE ROW CHECK
# =============================================================================
print("\n[6] DUPLICATE ROWS CHECK")
print("-" * 40)
# df.duplicated() returns True for any row that is an EXACT copy of a previous row
# .sum() counts how many such duplicates exist
# Duplicates can bias the model (it "memorises" repeated data instead of learning)
dup_count = df.duplicated().sum()
print(f"\n  Number of duplicate rows: {dup_count}")
if dup_count == 0:
    print("  RESULT: No duplicates found! Each student record is unique.")
else:
    print(f"  RESULT: {dup_count} duplicate rows found! Dropping them...")
    df = df.drop_duplicates()
    print(f"  Dataset shape after dropping: {df.shape}")

# =============================================================================
# SECTION 7: VISUALISATIONS
# =============================================================================
print("\n[7] GENERATING VISUALISATIONS...")
print("-" * 40)

FEATURES = ["study_hours", "attendance_pct", "prev_exam_score",
            "assignments_done", "sleep_hours"]
TARGET   = "final_score"

# ── PLOT 1: Distribution of Every Feature ─────────────────────────────────────
# A distribution plot (histogram) shows: how often does each value appear?
# - Tall bars → that value is common in the dataset
# - Short bars → that value is rare
# Why it matters: Unusual distributions (very skewed, outlier spikes) can
# affect how well a model learns.
print("    Saving Plot 1: Feature distributions...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Distribution of Each Feature & Target Variable",
             fontsize=16, fontweight="bold", y=1.01)

all_cols = FEATURES + [TARGET]
colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

for i, (col, ax, color) in enumerate(zip(all_cols, axes.flatten(), colors)):
    # histplot draws the bars (the actual count of students per value bucket)
    # kde=True overlays a smooth "density curve" — it's the smoothed shape of distribution
    sns.histplot(df[col], ax=ax, color=color, kde=True,
                 edgecolor="white", linewidth=0.5)
    ax.set_title(f"{col.replace('_', ' ').title()}", fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Number of Students")
    # A vertical dashed line showing the MEAN — "where most students cluster"
    ax.axvline(df[col].mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("notebooks/eda_plots/01_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("      --> Saved: notebooks/eda_plots/01_distributions.png")

# ── PLOT 2: Correlation Heatmap ───────────────────────────────────────────────
# Correlation measures how strongly two variables move TOGETHER.
# Range: -1 to +1
#   +1.0 = perfect positive relationship (as X goes up, Y goes up perfectly)
#    0.0 = no relationship at all (X tells us nothing about Y)
#   -1.0 = perfect negative relationship (as X goes up, Y goes down perfectly)
# In ML: high correlation with TARGET = that feature is likely very USEFUL!
print("    Saving Plot 2: Correlation heatmap...")

fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df.corr(numeric_only=True)  # compute correlation for all column pairs

# annot=True puts the number inside each cell so you can read exact values
# fmt=".2f means format as 2 decimal places (e.g., 0.75 not 0.748273)
# cmap="coolwarm": red = high positive correlation, blue = high negative
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Correlation Coefficient"})
ax.set_title("Correlation Matrix — All Features vs Target",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("notebooks/eda_plots/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("      --> Saved: notebooks/eda_plots/02_correlation_heatmap.png")

# Print a summary of each feature's correlation with final_score
print("\n  Correlation of each FEATURE with TARGET (final_score):")
target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)
for feat, val in target_corr.items():
    bar = "#" * int(abs(val) * 20)
    direction = "+" if val > 0 else "-"
    print(f"    {feat:<22} {direction}{abs(val):.3f}  {bar}")

# ── PLOT 3: Feature vs Target Scatter Plots ───────────────────────────────────
# A scatter plot puts EACH STUDENT as a dot.
# X-axis = one feature, Y-axis = final_score
# If the dots form an upward diagonal: strong positive relationship
# If the dots look like a random cloud: weak or no relationship
print("\n    Saving Plot 3: Feature vs Target scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Feature vs Final Score (Scatter Plots)",
             fontsize=16, fontweight="bold", y=1.01)

# We'll colour the dots by final_score using a gradient — darker = higher score
for i, (feat, ax) in enumerate(zip(FEATURES, axes.flatten())):
    scatter = ax.scatter(df[feat], df[TARGET],
                         c=df[TARGET],          # colour by final score
                         cmap="viridis",        # viridis: purple(low) → yellow(high)
                         alpha=0.5,             # 50% transparent so overlapping dots show
                         edgecolors="none",
                         s=20)                  # dot size

    # Add a trend line using numpy's polyfit (fits a straight line through the cloud)
    # This makes the overall direction of the relationship immediately visible
    z = np.polyfit(df[feat], df[TARGET], 1)   # fit degree-1 polynomial (straight line)
    p = np.poly1d(z)                           # turn coefficients into a function
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend line")

    ax.set_xlabel(feat.replace("_", " ").title())
    ax.set_ylabel("Final Score")
    ax.set_title(f"{feat.replace('_', ' ').title()} vs Final Score",
                 fontweight="bold")
    ax.legend(fontsize=9)

# Hide the unused 6th subplot (we have 5 features, grid is 2x3=6)
axes[1, 2].set_visible(False)
plt.colorbar(scatter, ax=axes[1, 2], label="Final Score")
axes[1, 2].set_visible(True)
axes[1, 2].set_axis_off()

plt.tight_layout()
plt.savefig("notebooks/eda_plots/03_feature_vs_target.png", dpi=150, bbox_inches="tight")
plt.close()
print("      --> Saved: notebooks/eda_plots/03_feature_vs_target.png")

# ── PLOT 4: Target Distribution (final_score) ─────────────────────────────────
print("    Saving Plot 4: Target variable distribution...")

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df[TARGET], kde=True, color="#4C72B0",
             edgecolor="white", linewidth=0.5, ax=ax)

# Annotate key statistics on the plot
ax.axvline(df[TARGET].mean(),   color="red",    linestyle="--", lw=2,
           label=f"Mean: {df[TARGET].mean():.1f}")
ax.axvline(df[TARGET].median(), color="green",  linestyle="--", lw=2,
           label=f"Median: {df[TARGET].median():.1f}")

ax.set_title("Distribution of Final Scores (Target Variable)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Final Score (0-100)")
ax.set_ylabel("Number of Students")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/eda_plots/04_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("      --> Saved: notebooks/eda_plots/04_target_distribution.png")

# ── PLOT 5: Box Plots — Spotting Outliers ─────────────────────────────────────
# A box plot shows the spread of data and flags OUTLIERS (unusual values).
# The box covers the middle 50% of data (25th–75th percentile).
# The line inside the box = median.
# The "whiskers" extend to min/max (within 1.5x the box width).
# Any dots BEYOND the whiskers = outliers — unusually high or low values.
print("    Saving Plot 5: Box plots (outlier detection)...")

fig, axes = plt.subplots(1, len(all_cols), figsize=(16, 5))
fig.suptitle("Box Plots — Spread and Outlier Detection",
             fontsize=14, fontweight="bold")

for col, ax, color in zip(all_cols, axes, colors):
    sns.boxplot(y=df[col], ax=ax, color=color, width=0.5,
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
    ax.set_title(col.replace("_", " ").title(), fontweight="bold")
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("notebooks/eda_plots/05_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("      --> Saved: notebooks/eda_plots/05_boxplots.png")

# =============================================================================
# SECTION 8: FINAL EDA SUMMARY
# =============================================================================
print("\n" + "=" * 62)
print("  EDA SUMMARY")
print("=" * 62)
print(f"  Total students  : {len(df)}")
print(f"  Total features  : {len(FEATURES)}")
print(f"  Missing values  : {df.isnull().sum().sum()}")
print(f"  Duplicate rows  : {df.duplicated().sum()}")
print(f"  Final score range: {df[TARGET].min()} - {df[TARGET].max()}")
print(f"  Final score mean : {df[TARGET].mean():.2f}")
print(f"  Final score std  : {df[TARGET].std():.2f}")
print(f"\n  Most correlated feature with final_score:")
print(f"    --> {target_corr.idxmax()} (r = {target_corr.max():.3f})")
print(f"\n  Least correlated feature with final_score:")
print(f"    --> {target_corr.idxmin()} (r = {target_corr.min():.3f})")
print("\n  Plots saved to: notebooks/eda_plots/")
print("=" * 62)
