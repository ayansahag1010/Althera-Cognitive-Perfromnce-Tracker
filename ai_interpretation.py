import pandas as pd
import ollama

# Load combined data
data = pd.read_csv("combined_data.csv")

# Get latest record
row = data.iloc[-1]

# Create prompt for AI
prompt = f"""
Analyze the following cognitive health data.

Heart Rate: {row['HeartRate']} bpm
SpO2: {row['SpO2']} %
Motion Level: {row['Motion']}

Reaction Time: {row['simple_reaction_ms']} ms
Finger Taps: {row['finger_taps']}

Word Recall Score: {row['word_recall_score']}
Number Recall Score: {row['number_recall_score']}
Stroop Accuracy: {row['stroop_accuracy_percent']} %

Provide:
1. Cognitive health summary
2. Possible concerns
3. Suggestions for improvement
"""

# Send prompt to Ollama
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

report = response["message"]["content"]

print("\nAI Cognitive Report:\n")
print(report)

# Save report
with open("ai_report.txt", "w") as f:
    f.write(report)