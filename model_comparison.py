"""
Model Comparison Script — Althera Cognitive Monitoring
======================================================
Compares three regression models on the existing dataset
using the same features and target as train_model.py.

Models:
  1. Random Forest Regressor  (existing model)
  2. Gradient Boosting Regressor
  3. Support Vector Regressor (RBF kernel)

Outputs:
  - Console table with R² and RMSE for each model
  - model_r2_comparison.png
  - model_rmse_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load dataset ─────────────────────────────────────────────
data = pd.read_csv("combined_data.csv")
data = data.fillna(0)

# ── 2. Feature set (same as predict_score.py / train_model.py) ──
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

# If dataset is too small for train/test split, augment with
# gaussian noise around the real values so the comparison is valid.
MIN_SAMPLES = 30
if len(data) < MIN_SAMPLES:
    print(f"⚠  Dataset has only {len(data)} row(s). Augmenting to "
          f"{200} samples using gaussian noise around real values.\n")
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

print(f"Dataset size : {len(data)} samples")
print(f"Training set : {len(X_train)} samples")
print(f"Test set     : {len(X_test)} samples")
print()

# ── 4. Define models ────────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),
    "Support Vector Regressor": SVR(
        kernel="rbf",
        C=100,
        gamma="scale",
        epsilon=0.1,
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

# ── 6. Print results table ──────────────────────────────────────
print("=" * 60)
print("            MODEL COMPARISON RESULTS")
print("=" * 60)
print(f"{'Model':<28} {'R² Score':>12} {'RMSE':>12}")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<28} {row['R2_Score']:>12.4f} {row['RMSE']:>12.4f}")
print("=" * 60)

best_r2 = results_df.loc[results_df["R2_Score"].idxmax()]
best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
print(f"\n★ Best R² Score : {best_r2['Model']} ({best_r2['R2_Score']:.4f})")
print(f"★ Best RMSE     : {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")

# ── 7. Generate comparison graphs ───────────────────────────────
# --- Style settings ---
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#3a3a5c",
    "axes.labelcolor": "#e0e0f0",
    "text.color": "#e0e0f0",
    "xtick.color": "#e0e0f0",
    "ytick.color": "#e0e0f0",
    "grid.color": "#2a2a4a",
    "font.family": "sans-serif",
    "font.size": 12,
})

COLORS = ["#7c3aed", "#06b6d4", "#f59e0b"]
model_names = results_df["Model"].tolist()

# ── R² Comparison Bar Chart ──
fig1, ax1 = plt.subplots(figsize=(9, 5))
bars1 = ax1.bar(model_names, results_df["R2_Score"], color=COLORS,
                edgecolor="#e0e0f0", linewidth=0.8, width=0.55)
for bar, val in zip(bars1, results_df["R2_Score"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontweight="bold",
             fontsize=11, color="#e0e0f0")
ax1.set_title("R² Score Comparison", fontsize=16, fontweight="bold", pad=15)
ax1.set_ylabel("R² Score", fontsize=13)
ax1.set_ylim(min(0, results_df["R2_Score"].min() - 0.1),
             max(1.05, results_df["R2_Score"].max() + 0.15))
ax1.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig1.savefig("model_r2_comparison.png", dpi=200, bbox_inches="tight")
print("\n✅ Saved: model_r2_comparison.png")

# ── RMSE Comparison Bar Chart ──
fig2, ax2 = plt.subplots(figsize=(9, 5))
bars2 = ax2.bar(model_names, results_df["RMSE"], color=COLORS,
                edgecolor="#e0e0f0", linewidth=0.8, width=0.55)
for bar, val in zip(bars2, results_df["RMSE"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontweight="bold",
             fontsize=11, color="#e0e0f0")
ax2.set_title("RMSE Comparison", fontsize=16, fontweight="bold", pad=15)
ax2.set_ylabel("RMSE", fontsize=13)
ax2.set_ylim(0, results_df["RMSE"].max() * 1.3)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig2.savefig("model_rmse_comparison.png", dpi=200, bbox_inches="tight")
print("✅ Saved: model_rmse_comparison.png")

print("\n🎯 Model comparison complete!")
