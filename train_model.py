import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load combined data
data = pd.read_csv("combined_data.csv")

# Replace missing values
data = data.fillna(0)

# Feature columns
features = data[[
    "HeartRate",
    "SpO2",
    "Motion",
    "simple_reaction_ms",
    "choice_reaction_ms",
    "finger_taps",
    "word_recall_score",
    "number_recall_score",
    "stroop_accuracy_percent"
]]

# Create a simple cognitive score formula
data["CognitiveScore"] = (
    100
    - (data["simple_reaction_ms"] * 0.05)
    + (data["word_recall_score"] * 5)
    + (data["stroop_accuracy_percent"] * 0.2)
)

target = data["CognitiveScore"]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(features, target)

# Save model
joblib.dump(model, "cognitive_model.pkl")

print("Model trained successfully.")