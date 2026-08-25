"""
Appends Stage 3 cells into the existing Jupyter notebook.
Run: python notebooks/append_stage3.py
"""
import json

NOTEBOOK_PATH = "notebooks/student_performance_prediction.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

stage3_cells = [
    # ── Section header ─────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Stage 3: Data Cleaning\n",
            "\n",
            "Even though our dataset is already clean, real-world data is almost always messy.\n",
            "We will deliberately inject problems into a copy of the data, then clean it step-by-step.\n",
            "\n",
            "### The 4 Most Common Data Problems\n",
            "\n",
            "| Problem | What it looks like | How to fix it |\n",
            "|---------|-------------------|---------------|\n",
            "| Missing values | `NaN` cells | Fill with median/mode, or drop the row |\n",
            "| Duplicate rows | Identical student records | `df.drop_duplicates()` |\n",
            "| Outliers | `study_hours = 99` | Clip to valid range or use IQR |\n",
            "| Wrong data types | Numbers stored as text | `df['col'].astype(float)` |"
        ]
    },
    # ── Create dirty dataset ────────────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- Simulate a DIRTY dataset ---\n",
            "# We inject 3 types of problems into a copy so we can practice fixing them.\n",
            "np.random.seed(99)\n",
            "df_dirty = df.copy()\n",
            "N = len(df_dirty)\n",
            "\n",
            "# Problem 1: inject 30 missing values (NaN) across 4 columns\n",
            "idx = np.random.choice(N, 30, replace=False)\n",
            "df_dirty.loc[idx[:10],   'study_hours']      = np.nan\n",
            "df_dirty.loc[idx[10:20], 'attendance_pct']   = np.nan\n",
            "df_dirty.loc[idx[20:25], 'sleep_hours']      = np.nan\n",
            "df_dirty.loc[idx[25:],   'assignments_done'] = np.nan\n",
            "\n",
            "# Problem 2: inject 12 exact duplicate rows\n",
            "df_dirty = pd.concat([df_dirty, df_dirty.iloc[5:17]], ignore_index=True)\n",
            "\n",
            "# Problem 3: inject 3 physically impossible outliers\n",
            "df_dirty.loc[500, 'study_hours']    = 99.0   # impossible\n",
            "df_dirty.loc[501, 'sleep_hours']    = 0.0    # impossible\n",
            "df_dirty.loc[502, 'attendance_pct'] = 150.0  # impossible\n",
            "\n",
            "print('Dirty dataset shape:', df_dirty.shape)\n",
            "print('\\nMissing values per column:')\n",
            "print(df_dirty.isnull().sum())"
        ]
    },
    # ── Step A: Fix missing values ──────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step A: Fix Missing Values\n",
            "\n",
            "**Why use MEDIAN (not mean) for continuous columns?**\n",
            "\n",
            "Mean is pulled by extreme values (outliers). Median is not.\n",
            "- Values: `[80, 85, 82, 99]` \u2192 Mean = 86.5, Median = 82.5\n",
            "- If 99 is an outlier, median gives a more realistic fill value.\n",
            "\n",
            "**Why use MODE for discrete/integer columns?**\n",
            "- Mode = the most frequently occurring value\n",
            "- Makes sense for whole-number counts like `assignments_done`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_fixed = df_dirty.copy()\n",
            "\n",
            "# Fill CONTINUOUS columns with MEDIAN (robust to outliers)\n",
            "for col in ['study_hours', 'attendance_pct', 'sleep_hours']:\n",
            "    median_val = df_fixed[col].median()\n",
            "    df_fixed[col] = df_fixed[col].fillna(median_val)\n",
            "    print(f'{col}: filled missing values with median = {median_val:.2f}')\n",
            "\n",
            "# Fill DISCRETE column with MODE (most common value)\n",
            "mode_val = df_fixed['assignments_done'].mode()[0]\n",
            "df_fixed['assignments_done'] = df_fixed['assignments_done'].fillna(mode_val)\n",
            "print(f'assignments_done: filled missing values with mode = {mode_val}')\n",
            "\n",
            "print(f'\\nMissing values remaining: {df_fixed.isnull().sum().sum()}')"
        ]
    },
    # ── Step B: Remove duplicates ───────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step B: Remove Duplicate Rows"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f'Shape before: {df_fixed.shape}')\n",
            "\n",
            "# drop_duplicates() removes rows that are exact copies of earlier rows\n",
            "# reset_index(drop=True) re-numbers the rows cleanly after removal\n",
            "df_fixed = df_fixed.drop_duplicates().reset_index(drop=True)\n",
            "\n",
            "print(f'Shape after removing duplicates: {df_fixed.shape}')"
        ]
    },
    # ── Step C: Handle outliers ─────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step C: Handle Outliers\n",
            "\n",
            "**Method 1 \u2014 IQR (Interquartile Range):**\n",
            "```\n",
            "IQR         = Q3 - Q1\n",
            "Lower fence = Q1 - 1.5 * IQR\n",
            "Upper fence = Q3 + 1.5 * IQR\n",
            "```\n",
            "Values outside these fences are statistical outliers.\n",
            "\n",
            "**Method 2 \u2014 Domain Knowledge (common sense):**\n",
            "- `study_hours > 24` \u2192 physically impossible\n",
            "- `attendance_pct > 100` \u2192 logically impossible\n",
            "- `sleep_hours = 0` \u2192 biologically impossible\n",
            "\n",
            "We use `df[col].clip(lower, upper)` to cap values at the valid boundary."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Domain-knowledge clipping: define min and max for each column\n",
            "domain_rules = {\n",
            "    'study_hours':      (0.0, 20.0),\n",
            "    'attendance_pct':   (0.0, 100.0),\n",
            "    'sleep_hours':      (1.0, 14.0),\n",
            "    'assignments_done': (0,   10),\n",
            "    'prev_exam_score':  (0.0, 100.0),\n",
            "    'final_score':      (0.0, 100.0),\n",
            "}\n",
            "\n",
            "for col, (lo, hi) in domain_rules.items():\n",
            "    n_out = ((df_fixed[col] < lo) | (df_fixed[col] > hi)).sum()\n",
            "    if n_out > 0:\n",
            "        df_fixed[col] = df_fixed[col].clip(lower=lo, upper=hi)\n",
            "        print(f'{col}: {n_out} outlier(s) clipped to [{lo}, {hi}]')\n",
            "\n",
            "print(f'\\nFinal shape after all cleaning: {df_fixed.shape}')"
        ]
    },
    # ── Problem Framing ─────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Stage 3: Problem Framing \u2014 Why Regression?\n",
            "\n",
            "### The Golden Rule\n",
            "\n",
            "> **Look at your TARGET variable. Ask: \"Is it a NUMBER or a CATEGORY?\"**\n",
            ">\n",
            "> - **Number (continuous)** \u2192 **REGRESSION**\n",
            "> - **Category (label)** \u2192 **CLASSIFICATION**\n",
            "\n",
            "### Our Case\n",
            "\n",
            "| Question | Answer |\n",
            "|----------|--------|\n",
            "| Target variable | `final_score` |\n",
            "| Sample values | 61.1, 83.2, 84.9, 89.9, 69.0 |\n",
            "| Type of value | A continuous decimal number between 0 and 100 |\n",
            "| Problem type | **REGRESSION** |\n",
            "\n",
            "### What Would Make It Classification Instead?\n",
            "\n",
            "| If we predicted... | Type |\n",
            "|--------------------|------|\n",
            "| Pass / Fail | Binary Classification (2 labels) |\n",
            "| Grade A / B / C / D / F | Multi-class Classification (5 labels) |\n",
            "| At-risk: Yes / No | Binary Classification |\n",
            "| **Actual score (0\u2013100)** | **Regression \u2190 our task** |\n",
            "\n",
            "### The Analogy\n",
            "- **Regression** = *\"HOW MUCH?\"* \u2192 \"What is the exact score?\"\n",
            "- **Classification** = *\"WHICH ONE?\"* \u2192 \"Which grade letter?\""
        ]
    },
    # ── Prepare X and y ─────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Stage 3: Prepare X and y for Modelling\n",
            "\n",
            "- **`X`** \u2014 the feature matrix: what the model sees as input (500 \u00d7 5)\n",
            "- **`y`** \u2014 the target vector: what the model must predict (500 values)\n",
            "\n",
            "We use the original clean dataset for modelling (not the messy demo version)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "FEATURES = ['study_hours', 'attendance_pct', 'prev_exam_score',\n",
            "            'assignments_done', 'sleep_hours']\n",
            "TARGET   = 'final_score'\n",
            "\n",
            "# X = input features only (we exclude final_score)\n",
            "# If we included final_score in X, the model would learn nothing useful\n",
            "# because it would just read the answer directly.\n",
            "X = df[FEATURES]\n",
            "y = df[TARGET]\n",
            "\n",
            "print(f'X shape: {X.shape}  -- rows=students, cols=features')\n",
            "print(f'y shape: {y.shape}  -- one score per student')\n",
            "print(f'\\nFirst 3 rows of X:')\n",
            "print(X.head(3))\n",
            "print(f'\\nFirst 3 values of y:')\n",
            "print(y.head(3).to_string())\n",
            "print('\\nStage 3 complete. Ready for Stage 4: Model Training!')"
        ]
    },
    # ── Closing ─────────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "*Next: Stage 4 \u2014 Model Training (Linear Regression, Decision Tree, Random Forest)*"
        ]
    }
]

# Remove the old Stage 2 closing markdown cell (last cell) and append Stage 3
nb["cells"] = nb["cells"][:-1] + stage3_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated successfully!")
print(f"Total cells now: {len(nb['cells'])}")
