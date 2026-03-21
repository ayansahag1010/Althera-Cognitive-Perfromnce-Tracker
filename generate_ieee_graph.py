"""
Generate IEEE-format model comparison graph.
Trains the same 3 models and produces a single combined bar chart
with R2 Score and RMSE side-by-side for each model.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load dataset ─────────────────────────────────────────────
data = pd.read_csv("combined_data.csv")
data = data.fillna(0)

# ── 2. Feature set (same as train_model.py) ─────────────────────
FEATURE_COLS = [
    "HeartRate",
    "SpO2",
    "Motion",
    "simple_reaction_ms",
    "choice_reaction_ms",
    "finger_taps",
    "word_recall_score",
    "number_recall_score",
    "stroop_accuracy_percent",
]

# Augment if dataset is too small
MIN_SAMPLES = 30
if len(data) < MIN_SAMPLES:
    np.random.seed(42)
    augmented_rows = []
    for _ in range(200):
        row = data.sample(n=1, random_state=None).iloc[0].copy()
        for col in FEATURE_COLS:
            noise = np.random.normal(0, abs(row[col]) * 0.05 + 1e-6)
            row[col] = row[col] + noise
        augmented_rows.append(row)
    data = pd.DataFrame(augmented_rows).reset_index(drop=True)
    data = data.fillna(0)

X = data[FEATURE_COLS]

# Target — same formula as train_model.py
data["CognitiveScore"] = (
    100
    - (data["simple_reaction_ms"] * 0.05)
    + (data["word_recall_score"] * 5)
    + (data["stroop_accuracy_percent"] * 0.2)
)
y = data["CognitiveScore"]

# ── 3. Train / Test split (80/20) ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Define models ────────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
    ),
    "Support Vector\nRegressor": SVR(
        kernel="rbf", C=100, gamma="scale", epsilon=0.1,
    ),
}

# ── 5. Train & Evaluate ─────────────────────────────────────────
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({"Model": name, "R2_Score": r2, "RMSE": rmse})

results_df = pd.DataFrame(results)

# Print results
print("=" * 60)
print("            MODEL COMPARISON RESULTS")
print("=" * 60)
print(f"{'Model':<28} {'R2 Score':>12} {'RMSE':>12}")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"{row['Model'].replace(chr(10), ' '):<28} {row['R2_Score']:>12.4f} {row['RMSE']:>12.4f}")
print("=" * 60)

# ── 6. Generate IEEE-format combined bar chart ──────────────────
# Reset all matplotlib params to defaults first
plt.rcdefaults()

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

model_names = results_df["Model"].tolist()
r2_values = results_df["R2_Score"].tolist()
rmse_values = results_df["RMSE"].tolist()

x = np.arange(len(model_names))
bar_width = 0.3

fig, ax = plt.subplots(figsize=(7, 4.5))

# Two bars per model: R2 Score (black) and RMSE (gray)
bars_r2 = ax.bar(
    x - bar_width / 2, r2_values,
    width=bar_width, color="black", edgecolor="black",
    linewidth=0.5, label="R2 Score",
)
bars_rmse = ax.bar(
    x + bar_width / 2, rmse_values,
    width=bar_width, color="#888888", edgecolor="black",
    linewidth=0.5, label="RMSE",
)

# Value annotations on top of each bar (plain font)
for bar, val in zip(bars_r2, r2_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="black",
    )
for bar, val in zip(bars_rmse, rmse_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="black",
    )

# Axis labels only — no title
ax.set_xlabel("Models", fontsize=11)
ax.set_ylabel("Metric Value", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=9)

# Y-axis range
y_max = max(max(r2_values), max(rmse_values))
ax.set_ylim(0, y_max * 1.25)

# Subtle grid on y-axis only
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
ax.set_axisbelow(True)

# Legend
ax.legend(loc="upper right", frameon=True, edgecolor="black",
          fontsize=9, fancybox=False)

plt.tight_layout()
fig.savefig("model_comparison_ieee.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("\nSaved: model_comparison_ieee.png (300 DPI, IEEE format)")
