"""
Appends Stage 5 cells to student_performance_prediction.ipynb
"""
import json

NOTEBOOK_PATH = "notebooks/student_performance_prediction.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

stage5_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Stage 5: Best Model Selection & Prediction Function\n",
            "\n",
            "In Stage 4, **Linear Regression** emerged as the winning model:\n",
            "- **R² = 0.8941** (explains 89.4% of score variation)\n",
            "- **RMSE = 4.0941** marks (average prediction error)\n",
            "\n",
            "Now we will:\n",
            "1. Save the best model to disk using `pickle` (`src/best_model.pkl`)\n",
            "2. Build a reusable prediction function with input validation\n",
            "3. Test it on 5 diverse student profiles"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pickle\n",
            "import os\n",
            "\n",
            "# Save the trained Linear Regression model to disk\n",
            "os.makedirs('../src', exist_ok=True)\n",
            "model_path = '../src/best_model.pkl'\n",
            "\n",
            "with open(model_path, 'wb') as f:\n",
            "    pickle.dump(lr_model, f)\n",
            "\n",
            "print(f'Best model saved to: {model_path}')\n",
            "print(f'File size: {os.path.getsize(model_path) / 1024:.1f} KB')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### The Prediction Function\n",
            "\n",
            "This function takes a student's 5 attributes, validates them, formats them\n",
            "into a DataFrame, calls the model, and clips the result to [0, 100]."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def predict_student_score(study_hours, attendance_pct, prev_exam_score,\n",
            "                          assignments_done, sleep_hours, model=lr_model):\n",
            "    \"\"\"\n",
            "    Predict final score for a student given their 5 features.\n",
            "    \"\"\"\n",
            "    # 1. Validation\n",
            "    assert 0 <= study_hours <= 24, 'study_hours must be [0, 24]'\n",
            "    assert 0 <= attendance_pct <= 100, 'attendance_pct must be [0, 100]'\n",
            "    assert 0 <= prev_exam_score <= 100, 'prev_exam_score must be [0, 100]'\n",
            "    assert 0 <= assignments_done <= 10, 'assignments_done must be [0, 10]'\n",
            "    assert 0 <= sleep_hours <= 24, 'sleep_hours must be [0, 24]'\n",
            "\n",
            "    # 2. DataFrame input\n",
            "    input_df = pd.DataFrame([{\n",
            "        'study_hours':       study_hours,\n",
            "        'attendance_pct':    attendance_pct,\n",
            "        'prev_exam_score':   prev_exam_score,\n",
            "        'assignments_done':  assignments_done,\n",
            "        'sleep_hours':       sleep_hours,\n",
            "    }])\n",
            "\n",
            "    # 3. Predict & Clip\n",
            "    raw_score = model.predict(input_df)[0]\n",
            "    final_score = float(np.clip(raw_score, 0.0, 100.0))\n",
            "    return round(final_score, 1)\n",
            "\n",
            "print('Prediction function ready!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test on 5 student profiles\n",
            "test_profiles = [\n",
            "    ('Star Performer',       9.0, 95.0, 88.0, 10, 7.5),\n",
            "    ('Average Student',      5.0, 74.0, 63.0,  5, 7.0),\n",
            "    ('Struggling Student',   1.5, 52.0, 38.0,  1, 5.0),\n",
            "    ('Sleep-Deprived Grinder',8.0, 90.0, 80.0, 8, 4.5),\n",
            "    ('Good Habits, Weak Base',7.0, 88.0, 45.0, 9, 8.0),\n",
            "]\n",
            "\n",
            "print(f\"{'Profile':<25} {'Study':>6} {'Attend':>7} {'Prev':>6} {'Asgn':>5} {'Sleep':>6} {'Predicted':>10}\")\n",
            "print('-' * 70)\n",
            "for name, sh, ap, ps, ad, slp in test_profiles:\n",
            "    sc = predict_student_score(sh, ap, ps, ad, slp)\n",
            "    print(f\"{name:<25} {sh:>6.1f} {ap:>6.1f}% {ps:>6.1f} {ad:>5} {slp:>6.1f} {sc:>9.1f}/100\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## End of Notebook — Full Pipeline Complete!\n",
            "\n",
            "1. **Dataset**: 500 records generated & verified  \n",
            "2. **EDA**: Visualised distributions, correlations, outliers  \n",
            "3. **Data Cleaning**: Imputed missing values, removed duplicates, clipped outliers  \n",
            "4. **Modelling**: Linear Regression (R²=0.894) > Random Forest (0.862) > Decision Tree (0.667)  \n",
            "5. **Prediction**: Serialised model to disk & built reusable `predict_score()` function  "
        ]
    }
]

nb["cells"] = nb["cells"][:-1] + stage5_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with Stage 5 cells!")
print(f"Total cells: {len(nb['cells'])}")
