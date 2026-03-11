import pandas as pd
import joblib
import pyttsx3

# Load trained model
model = joblib.load("cognitive_model.pkl")

# Load combined data
data = pd.read_csv("combined_data.csv")

# Replace missing values
data = data.fillna(0)

# Select features
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

# Predict cognitive score
score = model.predict(features)

print("\nPredicted Cognitive Health Score:", round(score[0], 2))

# Interpretation
if score[0] > 85:
    status = "Excellent cognitive performance"
elif score[0] > 70:
    status = "Normal cognitive performance"
elif score[0] > 50:
    status = "Mild cognitive fatigue"
else:
    status = "Low cognitive performance"

print("Status:", status)

# Voice feedback
engine = pyttsx3.init()

text = f"Your cognitive health score is {round(score[0],2)}. Status: {status}"

engine.say(text)
engine.runAndWait()