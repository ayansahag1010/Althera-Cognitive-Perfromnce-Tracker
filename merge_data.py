import pandas as pd

# Load CSV files
sensor = pd.read_csv("sensor_data.csv")
reaction = pd.read_csv("reaction_results.csv")
memory = pd.read_csv("memory_results.csv")

# Get latest rows
latest_sensor = sensor.tail(1).reset_index(drop=True)
latest_reaction = reaction.tail(1).reset_index(drop=True)
latest_memory = memory.tail(1).reset_index(drop=True)

# Merge them
combined = pd.concat([latest_sensor, latest_reaction, latest_memory], axis=1)

# Replace missing values
combined = combined.fillna(0)

# Save
combined.to_csv("combined_data.csv", index=False)

print("Combined data created:\n")
print(combined)