"""
Appends Stage 4 cells into the existing Jupyter notebook.
Run: python notebooks/append_stage4.py
"""
import json

NOTEBOOK_PATH = "notebooks/student_performance_prediction.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

stage4_cells = [
    # ── Section header ─────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Stage 4: Model Training\n",
            "\n",
            "We will train three different ML algorithms and compare their performance.\n",
            "\n",
            "| Model | Core Idea | Best For |\n",
            "|-------|-----------|----------|\n",
            "| Linear Regression | Fit a straight line through data | Simple, linear relationships |\n",
            "| Decision Tree | Build a tree of yes/no questions | Non-linear patterns, interpretable |\n",
            "| Random Forest | Average 100 different decision trees | Accuracy + robustness |"
        ]
    },
    # ── Imports ─────────────────────────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.linear_model    import LinearRegression\n",
            "from sklearn.tree            import DecisionTreeRegressor\n",
            "from sklearn.ensemble        import RandomForestRegressor\n",
            "from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score\n",
            "\n",
            "print('Scikit-learn imports complete!')"
        ]
    },
    # ── Train/test split explanation ────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 1: Train / Test Split\n",
            "\n",
            "**Why split the data at all?**\n",
            "\n",
            "If you train and test on the *same* data, the model just memorises the answers \u2014\n",
            "like re-reading a solved exam paper and acing it again. That tells you nothing\n",
            "about whether the model *learned* or just *memorised*.\n",
            "\n",
            "**Solution:** Hide 20% of the data from the model during training.\n",
            "Only use it at the very end to measure true performance.\n",
            "\n",
            "```\n",
            "Full dataset (500 students)\n",
            "       |\n",
            "       +------------ 80% = 400 students --> Training set (model learns here)\n",
            "       |\n",
            "       +------------ 20% = 100 students --> Test set (sealed envelope)\n",
            "```"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X,              # feature matrix\n",
            "    y,              # target vector\n",
            "    test_size=0.2,  # 20% of data goes to the test set\n",
            "    random_state=42 # seed: guarantees the same split every time you run this\n",
            ")\n",
            "\n",
            "print(f'Training set : {len(X_train)} students (80%) -- model learns from these')\n",
            "print(f'Test set     : {len(X_test)} students (20%) -- never seen during training')"
        ]
    },
    # ── Model 1: Linear Regression ──────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Model 1: Linear Regression\n",
            "\n",
            "**What it does:** Finds the best-fit straight line (or flat plane in 5D) through the data.\n",
            "\n",
            "It learns one **coefficient** (weight) per feature:\n",
            "```\n",
            "final_score = w1*study_hours + w2*attendance_pct + w3*prev_score\n",
            "            + w4*assignments + w5*sleep_hours + bias\n",
            "```\n",
            "\n",
            "**Strength:** Simple, fast, highly interpretable \u2014 you can explain exactly why a score changed.\n",
            "\n",
            "**Weakness:** Assumes relationships are perfectly linear. Cannot capture curves or U-shapes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lr_model = LinearRegression()\n",
            "# .fit() = training step: reads X_train & y_train, finds best-fit coefficients\n",
            "lr_model.fit(X_train, y_train)\n",
            "# .predict() = apply the learned line to generate predictions\n",
            "lr_preds = lr_model.predict(X_test)\n",
            "\n",
            "print('Learned coefficients (how much each feature moves the score):')\n",
            "for feat, coef in zip(FEATURES, lr_model.coef_):\n",
            "    print(f'  {feat:<22}: {coef:+.4f}')\n",
            "print(f'  bias (intercept)      : {lr_model.intercept_:.4f}')"
        ]
    },
    # ── Model 2: Decision Tree ──────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Model 2: Decision Tree Regressor\n",
            "\n",
            "**What it does:** Builds a tree of yes/no questions to narrow down a prediction.\n",
            "\n",
            "```\n",
            "Is study_hours > 6?\n",
            "   YES --> Is attendance_pct > 80?\n",
            "              YES --> Predict 88.5\n",
            "              NO  --> Predict 76.2\n",
            "   NO  --> Is prev_exam_score > 70?\n",
            "              YES --> Predict 68.4\n",
            "              NO  --> Predict 52.1\n",
            "```\n",
            "\n",
            "**Strength:** Captures non-linear patterns. Easy to visualise.\n",
            "\n",
            "**Weakness:** Without `max_depth`, it memorises training data (overfitting).\n",
            "`max_depth=10` limits tree to 10 levels, improving generalisation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)\n",
            "dt_model.fit(X_train, y_train)\n",
            "dt_preds = dt_model.predict(X_test)\n",
            "\n",
            "print(f'Tree depth     : {dt_model.get_depth()} levels')\n",
            "print(f'Number of leaves: {dt_model.get_n_leaves()} (each leaf = a final prediction)')"
        ]
    },
    # ── Model 3: Random Forest ──────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Model 3: Random Forest Regressor\n",
            "\n",
            "**What it does:** Builds 100 different Decision Trees, each trained on a random\n",
            "subset of rows and features. Final prediction = **average of all 100 trees**.\n",
            "\n",
            "**Analogy:** Instead of consulting one expert, consult 100 different experts\n",
            "and average their opinions. The crowd is wiser than any individual.\n",
            "\n",
            "**Why is averaging better?**\n",
            "Each tree makes different mistakes (random data/features). When averaged,\n",
            "those mistakes cancel out. This is called **ensemble learning**.\n",
            "\n",
            "**Strength:** Very accurate, handles non-linearity, resistant to overfitting.\n",
            "\n",
            "**Weakness:** Slower to train; predictions harder to interpret than a single tree."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rf_model = RandomForestRegressor(\n",
            "    n_estimators=100,  # build 100 decision trees\n",
            "    max_depth=10,      # each tree limited to 10 levels (prevents overfitting)\n",
            "    random_state=42    # reproducibility\n",
            ")\n",
            "rf_model.fit(X_train, y_train)\n",
            "rf_preds = rf_model.predict(X_test)\n",
            "\n",
            "print(f'Random Forest trained with {rf_model.n_estimators} trees!')"
        ]
    },
    # ── Metrics explanation + evaluation ────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 2: Evaluate All Three Models\n",
            "\n",
            "We use **4 standard regression metrics**:\n",
            "\n",
            "| Metric | Formula | Meaning | Better when |\n",
            "|--------|---------|---------|-------------|\n",
            "| **MAE** | avg(|actual - predicted|) | Average error in marks | Lower |\n",
            "| **MSE** | avg((actual - predicted)\u00b2) | Penalises large errors heavily | Lower |\n",
            "| **RMSE** | sqrt(MSE) | Typical error size, same unit as score | Lower |\n",
            "| **R\u00b2** | 1 - (SS_res / SS_tot) | % of variation explained by the model | Higher |\n",
            "\n",
            "**Quick intuition for R\u00b2:**\n",
            "- R\u00b2 = 0.90 \u2192 model explains 90% of why scores differ between students\n",
            "- R\u00b2 = 0.50 \u2192 model explains 50% (a coin flip is accounting for the rest!)\n",
            "- R\u00b2 = 1.00 \u2192 perfect predictions (unrealistic in practice)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def evaluate_model(name, y_true, y_pred):\n",
            "    mae  = mean_absolute_error(y_true, y_pred)\n",
            "    mse  = mean_squared_error(y_true, y_pred)\n",
            "    rmse = np.sqrt(mse)\n",
            "    r2   = r2_score(y_true, y_pred)\n",
            "    return {'Model': name, 'MAE': round(mae, 4), 'MSE': round(mse, 4),\n",
            "            'RMSE': round(rmse, 4), 'R2': round(r2, 4)}\n",
            "\n",
            "results = [\n",
            "    evaluate_model('Linear Regression', y_test, lr_preds),\n",
            "    evaluate_model('Decision Tree',     y_test, dt_preds),\n",
            "    evaluate_model('Random Forest',     y_test, rf_preds),\n",
            "]\n",
            "\n",
            "results_df = pd.DataFrame(results).set_index('Model')\n",
            "print('MODEL COMPARISON TABLE')\n",
            "print('=' * 60)\n",
            "print(results_df.to_string())\n",
            "print('=' * 60)\n",
            "\n",
            "best_name = results_df['R2'].idxmax()\n",
            "print(f'\\nWinner: {best_name} (highest R2, lowest RMSE)')"
        ]
    },
    # ── Visualization ───────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot: Metric Comparison Bar Charts"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model_labels = ['Linear\\nRegression', 'Decision\\nTree', 'Random\\nForest']\n",
            "bar_colors   = ['#4C72B0', '#DD8452', '#55A868']\n",
            "\n",
            "fig, axes = plt.subplots(1, 4, figsize=(16, 5))\n",
            "fig.suptitle('Model Comparison: All 4 Evaluation Metrics', fontsize=14, fontweight='bold')\n",
            "\n",
            "metric_map = [('MAE','lower=better'), ('MSE','lower=better'),\n",
            "              ('RMSE','lower=better'), ('R2','higher=better')]\n",
            "\n",
            "for (metric, label), ax in zip(metric_map, axes):\n",
            "    vals = [results_df.loc[m.replace('\\n',' '), metric] for m in model_labels]\n",
            "    best_idx = vals.index(min(vals)) if 'lower' in label else vals.index(max(vals))\n",
            "    bars = ax.bar(model_labels, vals, color=bar_colors,\n",
            "                  edgecolor='white', linewidth=1.5, width=0.5)\n",
            "    bars[best_idx].set_edgecolor('gold')\n",
            "    bars[best_idx].set_linewidth(3)\n",
            "    for bar, val in zip(bars, vals):\n",
            "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,\n",
            "                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')\n",
            "    ax.set_title(f'{metric}\\n({label})', fontweight='bold')\n",
            "    ax.set_ylim(0, max(vals) * 1.2)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot: Actual vs Predicted Scores\n",
            "\n",
            "Each dot = one test student. X-axis = their real score. Y-axis = model's prediction.\n",
            "\n",
            "**Perfect model** = all dots on the red diagonal line.\n",
            "**Scatter around the line** = prediction error. Less scatter = better model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "fig.suptitle('Actual vs Predicted Scores (Test Set)', fontsize=14, fontweight='bold')\n",
            "\n",
            "for ax, (name, preds, col) in zip(axes,\n",
            "    [('Linear Regression', lr_preds, '#4C72B0'),\n",
            "     ('Decision Tree',     dt_preds, '#DD8452'),\n",
            "     ('Random Forest',     rf_preds, '#55A868')]):\n",
            "    ax.scatter(y_test, preds, alpha=0.45, color=col, edgecolors='none', s=22)\n",
            "    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())\n",
            "    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')\n",
            "    r2   = results_df.loc[name, 'R2']\n",
            "    rmse = results_df.loc[name, 'RMSE']\n",
            "    ax.set_title(f'{name}\\nR2={r2:.3f}  RMSE={rmse:.2f}', fontweight='bold')\n",
            "    ax.set_xlabel('Actual Score')\n",
            "    ax.set_ylabel('Predicted Score')\n",
            "    ax.legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot: Random Forest Feature Importance\n",
            "\n",
            "Random Forest can report which features contributed most to its predictions.\n",
            "Higher importance = that feature was used more often and reduced error more."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "feat_imp = pd.DataFrame({\n",
            "    'Feature':    FEATURES,\n",
            "    'Importance': rf_model.feature_importances_\n",
            "}).sort_values('Importance', ascending=False)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(9, 4))\n",
            "bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'],\n",
            "               color=bar_colors[:5], edgecolor='white', height=0.55)\n",
            "for bar, val in zip(bars, feat_imp['Importance']):\n",
            "    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,\n",
            "            f'{val:.3f} ({val*100:.1f}%)', va='center', fontweight='bold')\n",
            "ax.set_xlabel('Feature Importance')\n",
            "ax.set_title('Random Forest: Which Features Matter Most?', fontweight='bold')\n",
            "ax.set_xlim(0, feat_imp['Importance'].max() * 1.35)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('\\nFeature importance confirms:')\n",
            "print(f'  Most important: {feat_imp.iloc[0][\"Feature\"]} ({feat_imp.iloc[0][\"Importance\"]:.1%})')\n",
            "print(f'  Least important: {feat_imp.iloc[-1][\"Feature\"]} ({feat_imp.iloc[-1][\"Importance\"]:.1%})')"
        ]
    },
    # ── Closing ─────────────────────────────────────────────────────────
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "*Next: Stage 5 \u2014 Best Model Selection + Prediction Function*"
        ]
    }
]

# Remove old closing markdown cell and append Stage 4
nb["cells"] = nb["cells"][:-1] + stage4_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated with Stage 4 cells!")
print(f"Total cells now: {len(nb['cells'])}")
