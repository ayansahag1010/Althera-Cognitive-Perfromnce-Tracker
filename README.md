<div align="center">

# 🧠 Althera — Cognitive Performance Tracker

### AI-Powered Multimodal Cognitive Health Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A real-time cognitive health monitoring platform designed for early detection of cognitive decline and Alzheimer's indicators through multimodal data fusion, machine learning, and generative AI.*

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Dashboard Workflow](#-dashboard-workflow)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Comparison](#-model-comparison)
- [Tech Stack](#-tech-stack)
- [Authors](#-authors)
- [License](#-license)

---

## 🔍 Overview

**Althera** integrates five data modalities to produce a comprehensive cognitive health assessment:

| Modality | Source | Metrics |
|:---|:---|:---|
| **Physiological** | ESP8266 + MAX30102 + MPU6050 | Heart Rate, SpO2, Motion |
| **Motor Function** | Interactive Tests | Reaction Time, Finger Tapping Speed |
| **Memory** | Interactive Tests | Word Recall, Number Recall |
| **Executive Function** | Stroop Test | Accuracy, Response Time |
| **Emotional State** | Webcam + OpenCV | Facial Emotion Classification |

All data flows into a **Random Forest** regression model that outputs a **Cognitive Health Score (0–100)**, followed by a **Generative AI interpretation** powered by Ollama (LLaMA 3.2).

---

## ✨ Key Features

- 🔗 **Real-Time Sensor Integration** — Live heart rate, SpO2, and motion data via ESP8266 serial connection
- ⚡ **Reaction Time Assessment** — Simple reaction, choice reaction, and finger tapping tests
- 🧩 **Memory & Cognitive Tests** — Word recall, number recall, and Stroop interference test
- 😊 **Facial Emotion Detection** — Webcam-based emotion analysis using OpenCV Haar Cascades
- 🤖 **ML Cognitive Scoring** — Random Forest model trained on 9 multimodal features
- 🧬 **AI-Generated Reports** — Detailed health interpretation via Ollama (LLaMA 3.2)
- 📊 **Interactive Dashboard** — 8-step guided Streamlit workflow with Plotly visualizations
- 🔊 **Voice Feedback** — Text-to-speech cognitive score announcement
- 📡 **Simulated Mode** — Full testing capability without physical hardware

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALTHERA DASHBOARD (Streamlit)                 │
├─────────────┬─────────────┬──────────────┬─────────────────────┤
│  Sensor     │  Cognitive   │   Emotion    │   Results &         │
│  Panel      │  Tests       │   Detection  │   Visualization     │
├─────────────┴─────────────┴──────────────┴─────────────────────┤
│                     BRIDGE LAYER (dashboard_bridge.py)           │
├─────────────┬─────────────┬──────────────┬─────────────────────┤
│ serial_     │ reaction_   │ emotion_     │ ai_                 │
│ reader.py   │ Test.py     │ detection.py │ interpretation.py   │
├─────────────┴─────────────┴──────────────┴─────────────────────┤
│                     ML PIPELINE                                  │
│          train_model.py → cognitive_model.pkl → predict_score.py │
├──────────────────────────────────────────────────────────────────┤
│                     DATA LAYER                                   │
│    sensor_data.csv + reaction_results.csv + memory_results.csv   │
│                    → combined_data.csv                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Althera-Cognitive-Perfromnce-Tracker/
│
├── dashboard_app.py            # Main Streamlit dashboard (entry point)
├── dashboard_bridge.py         # Bridge connecting dashboard to backend modules
├── dashboard_helpers.py        # Utility functions for the dashboard
├── dashboard_styles.py         # CSS styles & Plotly dark theme
├── dashboard.py                # Dashboard launcher script
│
├── serial_reader.py            # Reads live sensor data from ESP8266 via serial
├── import serial.py            # Alternate serial communication script
│
├── reaction_Test.py            # Terminal-based reaction time tests
├── memory_Test.py              # Terminal-based memory & Stroop tests
├── emotion_detection.py        # Webcam emotion detection using OpenCV
│
├── train_model.py              # Trains Random Forest on combined data
├── predict_score.py            # Predicts cognitive score with voice feedback
├── model_comparison.py         # Compares ML models (RF, SVR, GB, etc.)
│
├── merge_data.py               # Merges sensor + test data into combined CSV
├── save_results.py             # Saves test results to CSV
├── ai_interpretation.py        # Sends data to Ollama LLM for analysis
│
├── generate_ieee_graph.py      # Generates IEEE-format publication graphs
├── generate_ieee_graphs.py     # Extended IEEE graph generation
│
├── combined_data.csv           # Merged multimodal dataset
├── sensor_data.csv             # Raw sensor readings
├── reaction_results.csv        # Reaction test results
├── memory_results.csv          # Memory test results
│
├── model_comparison_ieee.png   # IEEE-format model comparison chart
├── r2_comparison_ieee.png      # R² score comparison chart
├── rmse_comparison_ieee.png    # RMSE comparison chart
├── model_r2_comparison.png     # R² comparison visualization
├── model_rmse_comparison.png   # RMSE comparison visualization
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Purpose |
|:---|:---|
| Python 3.8+ | Runtime |
| ESP8266 + MAX30102 + MPU6050 | Hardware sensors *(optional — simulated mode available)* |
| Ollama | Generative AI reports — [Install Ollama](https://ollama.com) |
| Webcam | Facial emotion detection |

### Installation

```bash
# Clone the repository
git clone https://github.com/ayansahag1010/Althera-Cognitive-Perfromnce-Tracker.git
cd Althera-Cognitive-Perfromnce-Tracker

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
pip install streamlit plotly numpy

# Pull the LLaMA model (for AI reports)
ollama pull llama3.2
```

### Run the Dashboard

```bash
streamlit run dashboard_app.py
```

---

## 🖥️ Dashboard Workflow

The dashboard guides you through an **8-step cognitive assessment pipeline**:

| Step | Module | Description |
|:---:|:---|:---|
| 1 | 🔗 Connect Sensor | Connect ESP8266 hardware or activate simulated data |
| 2 | 📡 Collect Data | Stream real-time heart rate, SpO2, and motion data |
| 3 | ⚡ Reaction Tests | Simple reaction, choice reaction, and finger tapping |
| 4 | 🧩 Memory Tests | Word recall, number recall, and Stroop test |
| 5 | 😊 Emotion Detection | Webcam-based facial emotion analysis |
| 6 | 🤖 ML Prediction | Random Forest cognitive score prediction |
| 7 | 🧬 AI Interpretation | LLaMA 3.2 generates detailed health report |
| 8 | 📊 Final Results | Comprehensive dashboard with score, gauges, and charts |

---

## 🤖 Machine Learning Pipeline

### Feature Set (9 Multimodal Features)

```
Physiological:  Heart Rate  |  SpO2  |  Motion Level
Motor Function: Simple Reaction Time  |  Choice Reaction Time  |  Finger Taps
Memory:         Word Recall Score  |  Number Recall Score
Executive:      Stroop Accuracy (%)
```

### Cognitive Score Formula

```python
CognitiveScore = 100 - (simple_reaction_ms × 0.05) + (word_recall_score × 5) + (stroop_accuracy × 0.2)
```

### Score Interpretation

| Score | Status | Indicator |
|:---:|:---|:---|
| **> 85** | 🟢 Excellent | Healthy cognitive performance |
| **70 – 85** | 🔵 Normal | Within expected range |
| **50 – 70** | 🟡 Mild Fatigue | Early signs of cognitive fatigue |
| **< 50** | 🔴 Low | Potential cognitive decline indicator |

---

## 📊 Model Comparison

Multiple regression models were evaluated for cognitive score prediction accuracy:

- **Random Forest Regressor** *(selected)*
- Support Vector Regression (SVR)
- Gradient Boosting Regressor
- Linear Regression
- K-Nearest Neighbors

IEEE-formatted comparison charts are included in the repository:

| Chart | File |
|:---|:---|
| Combined Model Comparison | `model_comparison_ieee.png` |
| R² Score Comparison | `r2_comparison_ieee.png` |
| RMSE Comparison | `rmse_comparison_ieee.png` |

---

## 🛠️ Tech Stack

| Category | Technology |
|:---|:---|
| **Language** | Python 3.8+ |
| **Dashboard** | Streamlit, Plotly |
| **Machine Learning** | scikit-learn (Random Forest, SVR, Gradient Boosting) |
| **Computer Vision** | OpenCV (Haar Cascade Classifiers) |
| **Generative AI** | Ollama (LLaMA 3.2) |
| **Hardware** | ESP8266, MAX30102 (Heart Rate/SpO2), MPU6050 (Accelerometer) |
| **Voice Output** | pyttsx3 |
| **Serial Comm** | pyserial |
| **Data Processing** | pandas, NumPy |

---

## 📋 Dependencies

```txt
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

**Ayan Saha** — [@ayansahag1010](https://github.com/ayansahag1010)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with ❤️ as a Minor Project*

</div>
