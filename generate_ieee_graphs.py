"""
IEEE-format model comparison graphs.
Trains the same 3 models on combined_data.csv and produces two separate
publication-ready bar charts for R2 Score and RMSE.
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
    "Random\nForest": RandomForestRegressor(
        n_estimators=100, random_state=42,
    ),
    "Gradient\nBoosting": GradientBoostingRegressor(
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
    label = row["Model"].replace("\n", " ")
    print(f"{label:<28} {row['R2_Score']:>12.4f} {row['RMSE']:>12.4f}")
print("=" * 60)

# ── 6. IEEE style settings ──────────────────────────────────────
plt.rcdefaults()
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

model_names = results_df["Model"].tolist()
r2_values = results_df["R2_Score"].tolist()
rmse_values = results_df["RMSE"].tolist()
x = np.arange(len(model_names))

BAR_COLOR = "#333333"
IEEE_FIGSIZE = (3.5, 2.5)

# ── GRAPH 1: R2 Score Comparison ────────────────────────────────
fig1, ax1 = plt.subplots(figsize=IEEE_FIGSIZE)

bars1 = ax1.bar(x, r2_values, width=0.5, color=BAR_COLOR,
                edgecolor="black", linewidth=0.4)

for bar, val in zip(bars1, r2_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7)

ax1.set_xlabel("Models", fontsize=8)
ax1.set_ylabel("R2 Score", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=7)
ax1.set_ylim(0, max(r2_values) * 1.15)
ax1.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.5, color="gray")
ax1.set_axisbelow(True)

plt.tight_layout()
fig1.savefig("r2_comparison_ieee.png", dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
print("\nSaved: r2_comparison_ieee.png")

# ── GRAPH 2: RMSE Comparison ───────────────────────────────────
fig2, ax2 = plt.subplots(figsize=IEEE_FIGSIZE)

bars2 = ax2.bar(x, rmse_values, width=0.5, color=BAR_COLOR,
                edgecolor="black", linewidth=0.4)

for bar, val in zip(bars2, rmse_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=7)

ax2.set_xlabel("Models", fontsize=8)
ax2.set_ylabel("RMSE", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=7)
ax2.set_ylim(0, max(rmse_values) * 1.2)
ax2.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.5, color="gray")
ax2.set_axisbelow(True)

plt.tight_layout()
fig2.savefig("rmse_comparison_ieee.png", dpi=300, bbox_inches="tight",
             facecolor="white", edgecolor="none")
print("Saved: rmse_comparison_ieee.png")

print("\nDone — both IEEE graphs generated at 300 DPI.")
