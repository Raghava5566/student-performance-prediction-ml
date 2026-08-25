"""
generate_dataset.py
-------------------
Creates a realistic synthetic dataset of 500 student records.

Why synthetic?
  - Real student data has privacy constraints.
  - We can control the patterns to make them realistic and learnable.
  - The relationships we embed mirror real-world observations:
      * More study hours  → tends to raise scores
      * Higher attendance → tends to raise scores
      * Better past score → tends to raise scores
      * More assignments  → tends to raise scores
      * Extreme sleep (too little OR too much) → slightly lowers performance

Formula for final_score (before clipping):
  base = 10
  + study_hours       * 3.5
  + attendance_pct    * 0.3
  + prev_exam_score   * 0.25
  + assignments_done  * 1.5
  + sleep_bonus       (peaks at 7 hrs, drops off on either side)
  + random noise      (real life isn't perfectly predictable!)

Then we clip to [0, 100] so no score goes below 0 or above 100.
"""

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
# Setting a "seed" means every time you run this script you get the EXACT same
# random numbers. Without it, you'd get different data each run — bad for
# reproducibility (a core principle in ML).
np.random.seed(42)

N = 500  # number of student records to generate

# ── Generate each feature independently ──────────────────────────────────────

# study_hours: daily hours spent studying
# np.random.uniform(low, high, N) → N random numbers uniformly spread between low and high
study_hours = np.random.uniform(0.5, 10.0, N).round(1)

# attendance_pct: % of classes attended (50% → 100%)
attendance_pct = np.random.uniform(50.0, 100.0, N).round(1)

# prev_exam_score: score in the previous exam (30 → 95)
prev_exam_score = np.random.uniform(30.0, 95.0, N).round(1)

# assignments_done: number of assignments submitted out of 10
# np.random.randint(low, high+1, N) → N random whole numbers between low and high (inclusive)
assignments_done = np.random.randint(0, 11, N)

# sleep_hours: average nightly sleep (4 → 10 hours)
sleep_hours = np.random.uniform(4.0, 10.0, N).round(1)

# ── Sleep bonus: peaks at 7 hours ────────────────────────────────────────────
# This models reality: too little sleep → bad; too much → also slightly bad
# The formula -(sleep - 7)^2 * 0.5 creates an inverted U-shape:
#   sleep=7  → bonus = 0     (no penalty)
#   sleep=4  → bonus = -4.5  (penalty for under-sleeping)
#   sleep=10 → bonus = -4.5  (penalty for over-sleeping)
sleep_bonus = -(sleep_hours - 7) ** 2 * 0.5

# ── Random noise ─────────────────────────────────────────────────────────────
# Real life isn't a perfect formula! Students have good days, bad days, test
# anxiety, family issues etc. We add Gaussian (bell-curve) noise to simulate this.
# loc=0 means centered around 0 (noise is as likely to be positive as negative)
# scale=4 means most noise values fall within ±4 points
noise = np.random.normal(loc=0, scale=4, size=N)

# ── Compute the final score ───────────────────────────────────────────────────
final_score = (
    10                          # base score everyone starts with
    + study_hours * 3.5         # each study hour adds ~3.5 points
    + attendance_pct * 0.3      # each attendance % point adds 0.3 points
    + prev_exam_score * 0.25    # past performance has moderate influence
    + assignments_done * 1.5    # each assignment adds 1.5 points
    + sleep_bonus               # sleep quality effect (inverted U)
    + noise                     # random real-life variation
)

# Clip: ensure no score is below 0 or above 100
final_score = np.clip(final_score, 0, 100).round(1)

# ── Assemble into a DataFrame ─────────────────────────────────────────────────
# A DataFrame is like an Excel spreadsheet inside Python (from the pandas library)
df = pd.DataFrame({
    "study_hours":       study_hours,
    "attendance_pct":    attendance_pct,
    "prev_exam_score":   prev_exam_score,
    "assignments_done":  assignments_done,
    "sleep_hours":       sleep_hours,
    "final_score":       final_score,
})

# ── Save to CSV ───────────────────────────────────────────────────────────────
output_path = "student_performance.csv"
df.to_csv(output_path, index=False)
# index=False → don't write the row numbers (0,1,2...) as a column in the CSV

print(f"[OK] Dataset saved to '{output_path}'")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()
print("── First 10 rows (df.head(10)) ──────────────────────────────")
print(df.head(10).to_string(index=True))
print()
print("── Dataset statistics (df.describe()) ──────────────────────")
print(df.describe().round(2).to_string())
