# 🧠 Althera — Multimodal Cognitive Health Monitoring System

**Althera** is an AI-powered cognitive health monitoring platform designed for early detection of cognitive decline and Alzheimer's indicators. It integrates real-time physiological sensors, interactive cognitive tests, facial emotion detection, machine learning prediction, and generative AI interpretation — all within a unified Streamlit dashboard.

---

## ✨ Features

| Module | Description |
|---|---|
| **Hardware Sensor Integration** | Real-time heart rate, SpO2, and motion data via ESP8266 + MAX30102 + MPU6050 |
| **Reaction Tests** | Simple reaction time, choice reaction, and finger tapping assessments |
| **Memory Tests** | Word recall, number recall, and Stroop test for cognitive evaluation |
| **Emotion Detection** | Webcam-based facial emotion analysis using OpenCV Haar Cascades |
| **ML Prediction** | Random Forest model predicts a cognitive health score from multimodal data |
| **AI Interpretation** | Generative AI report via Ollama (LLaMA 3.2) with health summary and recommendations |
| **Interactive Dashboard** | 8-step guided workflow with dark-themed Streamlit UI and Plotly visualizations |

---

## 🏗️ Project Structure

```
Althera-Minor-Project/
├── dashboard_app.py          # Main Streamlit dashboard (entry point)
├── dashboard_bridge.py       # Bridge module connecting dashboard to backend modules
├── dashboard_helpers.py      # Helper functions for the dashboard
├── dashboard_styles.py       # CSS styles and Plotly theme for the dashboard
├── dashboard.py              # Dashboard launcher
│
├── serial_reader.py          # Reads live sensor data from ESP8266 via serial port
├── import serial.py          # Alternate serial communication script
│
├── reaction_Test.py          # Standalone reaction time tests (terminal-based)
├── memory_Test.py            # Standalone memory and Stroop tests (terminal-based)
│
├── emotion_detection.py      # Webcam-based emotion detection using OpenCV
│
├── train_model.py            # Trains Random Forest model on combined data
├── predict_score.py          # Predicts cognitive score using the trained model
├── model_comparison.py       # Compares ML models (Random Forest, SVR, etc.)
│
├── merge_data.py             # Merges sensor + test data into combined_data.csv
├── save_results.py           # Saves test results to CSV files
├── ai_interpretation.py      # Sends data to Ollama LLM for cognitive analysis
│
├── generate_ieee_graph.py    # Generates IEEE-format publication graphs
├── generate_ieee_graphs.py   # Extended IEEE graph generation script
│
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Hardware (optional):** ESP8266 + MAX30102 (heart rate/SpO2) + MPU6050 (accelerometer)
- **Ollama** (for AI interpretation) — [Install Ollama](https://ollama.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayansahag1010/Althera-Minor-Project.git
   cd Althera-Minor-Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install streamlit plotly
   ```

4. **Pull the LLaMA model (for AI reports)**
   ```bash
   ollama pull llama3.2
   ```

### Running the Dashboard

```bash
streamlit run dashboard_app.py
```

The dashboard guides you through an **8-step workflow**:

1. 🔗 Connect Sensor (or use simulated data)
2. 📡 Collect Sensor Data
3. ⚡ Reaction Tests
4. 🧩 Memory & Cognitive Tests
5. 😊 Facial Emotion Detection
6. 🤖 ML Cognitive Score Prediction
7. 🧬 AI Interpretation Report
8. 📊 Final Results Dashboard

---

## 🔬 How It Works

### Data Collection
- **Physiological data** (heart rate, SpO2, motion) is collected via an ESP8266 microcontroller connected to MAX30102 and MPU6050 sensors
- **Cognitive tests** measure reaction time, memory recall, and executive function (Stroop test)
- **Emotion detection** uses OpenCV face detection to analyze facial expressions via webcam

### ML Prediction
A **Random Forest Regressor** is trained on 9 multimodal features:
- Heart Rate, SpO2, Motion Level
- Simple Reaction Time, Choice Reaction Time, Finger Taps
- Word Recall Score, Number Recall Score, Stroop Accuracy

The model outputs a **Cognitive Health Score (0–100)** with status classification:
| Score Range | Status |
|---|---|
| > 85 | Excellent cognitive performance |
| 70 – 85 | Normal cognitive performance |
| 50 – 70 | Mild cognitive fatigue |
| < 50 | Low cognitive performance |

### AI Interpretation
Collected data is sent to **Ollama (LLaMA 3.2)** which generates a detailed report including:
- Cognitive health summary
- Possible concerns
- Suggestions for improvement

---

## 📊 Model Comparison

Multiple regression models were evaluated for cognitive score prediction:

| Model | R² Score | RMSE |
|---|---|---|
| Random Forest | Best | Lowest |
| Support Vector Regression | — | — |
| Gradient Boosting | — | — |

IEEE-formatted comparison graphs are available in the project root.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Dashboard:** Streamlit + Plotly
- **ML:** scikit-learn (Random Forest, SVR, Gradient Boosting)
- **Computer Vision:** OpenCV (Haar Cascades)
- **AI/LLM:** Ollama (LLaMA 3.2)
- **Hardware:** ESP8266 + MAX30102 + MPU6050
- **Voice:** pyttsx3 (text-to-speech feedback)
- **Serial:** pyserial (sensor communication)

---

## 📋 Dependencies

```
pandas
opencv-python
pyserial
pyttsx3
scikit-learn
joblib
ollama
streamlit
plotly
numpy
```

---

## 👥 Authors

- **Ayan Saha** — [GitHub](https://github.com/ayansahag1010)

---

## 📄 License

This project is part of a Minor Project submission. All rights reserved.
