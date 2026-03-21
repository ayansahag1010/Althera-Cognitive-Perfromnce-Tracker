"""
Althera Dashboard Bridge Module
================================
Re-implements core logic from existing project modules for web dashboard integration.
Does NOT modify any existing files — all logic is mirrored here for Streamlit compatibility.

Wraps:
  - import serial.py  → SensorReader class
  - predict_score.py  → predict_cognitive_score()
  - ai_interpretation.py → generate_ai_report()
  - emotion_detection.py → detect_emotion_from_frame()
"""

import threading
import time
import os
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import joblib
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1. SENSOR READER  (mirrors import serial.py)
# ──────────────────────────────────────────────────────────────

class SensorReader:
    """
    Background thread that reads sensor data from an ESP8266/Arduino
    over serial.  Auto-detects the COM port and streams data into
    an in-memory deque (no CSV dependency).
    """

    def __init__(self, baud_rate=115200, buffer_size=500):
        self.baud_rate = baud_rate
        self.buffer = deque(maxlen=buffer_size)
        self.latest = {}
        self.connected = False
        self.port_name = None
        self._thread = None
        self._stop_event = threading.Event()
        self._serial = None

    # ---- auto-detect serial port ----
    @staticmethod
    def find_serial_port():
        """Return the first likely sensor COM port, or None."""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                desc = (p.description or "").lower()
                if any(kw in desc for kw in ("ch340", "cp210", "usb", "serial", "arduino", "esp")):
                    return p.device
            # fallback: return first available port
            if ports:
                return ports[0].device
        except Exception:
            pass
        return None

    # ---- start / stop ----
    def start(self, port=None):
        """Begin reading in a background thread."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self.port_name = port or self.find_serial_port()
        if not self.port_name:
            raise ConnectionError("No serial port detected. Please connect the sensor.")
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        self.connected = False

    # ---- internal read loop (mirrors import serial.py logic) ----
    def _read_loop(self):
        import serial as pyserial
        try:
            self._serial = pyserial.Serial(self.port_name, self.baud_rate, timeout=1)
            time.sleep(2)  # allow MCU reset
            self.connected = True

            while not self._stop_event.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()

                # Same parsing logic as import serial.py
                if not line.startswith("AX"):
                    continue

                parts = line.split(",")
                data = {}
                for item in parts:
                    if ":" in item:
                        key, value = item.split(":", 1)
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                record = {
                    "Timestamp": timestamp,
                    "AX": data.get("AX", 0),
                    "AY": data.get("AY", 0),
                    "AZ": data.get("AZ", 0),
                    "Motion": data.get("Motion", 0),
                    "HeartRate": data.get("HR", 0),
                    "SpO2": data.get("SpO2", 0),
                }
                self.buffer.append(record)
                self.latest = record

        except Exception as e:
            self.connected = False
            self.latest = {"error": str(e)}
        finally:
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self.connected = False

    # ---- convenience ----
    def get_latest(self):
        return dict(self.latest)

    def get_buffer_df(self):
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer))

    def get_averages(self):
        df = self.get_buffer_df()
        if df.empty:
            return {"HeartRate": 0, "SpO2": 0, "Motion": 0}
        return {
            "HeartRate": round(df["HeartRate"].astype(float).mean(), 1),
            "SpO2": round(df["SpO2"].astype(float).mean(), 1),
            "Motion": round(df["Motion"].astype(float).mean(), 2),
        }


# ──────────────────────────────────────────────────────────────
# 2. ML PREDICTION  (mirrors predict_score.py + train_model.py)
# ──────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cognitive_model.pkl")

FEATURE_COLUMNS = [
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


def predict_cognitive_score(data_dict: dict) -> tuple:
    """
    Load the trained RandomForest model and predict a cognitive score.

    Parameters
    ----------
    data_dict : dict
        Must contain keys matching FEATURE_COLUMNS.

    Returns
    -------
    (score: float, status: str)
    """
    model = joblib.load(MODEL_PATH)
    row = {col: float(data_dict.get(col, 0)) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    score = float(model.predict(df)[0])
    score = round(score, 2)

    # Same interpretation thresholds as predict_score.py
    if score > 85:
        status = "Excellent cognitive performance"
    elif score > 70:
        status = "Normal cognitive performance"
    elif score > 50:
        status = "Mild cognitive fatigue"
    else:
        status = "Low cognitive performance — further evaluation recommended"

    return score, status


# ──────────────────────────────────────────────────────────────
# 3. AI INTERPRETATION  (mirrors ai_interpretation.py)
# ──────────────────────────────────────────────────────────────

def generate_ai_report(data_dict: dict) -> str:
    """
    Send patient data to Ollama (llama3.2) and return a natural-language
    cognitive health report.  Same prompt template as ai_interpretation.py.
    """
    prompt = f"""
Analyze the following cognitive health data.

Heart Rate: {data_dict.get('HeartRate', 'N/A')} bpm
SpO2: {data_dict.get('SpO2', 'N/A')} %
Motion Level: {data_dict.get('Motion', 'N/A')}

Reaction Time: {data_dict.get('simple_reaction_ms', 'N/A')} ms
Finger Taps: {data_dict.get('finger_taps', 'N/A')}

Word Recall Score: {data_dict.get('word_recall_score', 'N/A')}
Number Recall Score: {data_dict.get('number_recall_score', 'N/A')}
Stroop Accuracy: {data_dict.get('stroop_accuracy_percent', 'N/A')} %

Predicted Cognitive Score: {data_dict.get('cognitive_score', 'N/A')}
Status: {data_dict.get('cognitive_status', 'N/A')}

Provide:
1. Cognitive health summary
2. Possible concerns
3. Suggestions for improvement
"""
    try:
        import ollama
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        return (
            f"⚠️ AI report generation failed: {e}\n\n"
            "Make sure Ollama is running (`ollama serve`) and the "
            "`llama3.2` model is pulled (`ollama pull llama3.2`)."
        )


# ──────────────────────────────────────────────────────────────
# 4. EMOTION DETECTION  (mirrors emotion_detection.py)
# ──────────────────────────────────────────────────────────────

# Load cascade once at module level
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_emotion_from_frame(frame):
    """
    Run face detection + rule-based emotion classification on a single
    BGR frame.  Same logic as emotion_detection.py.

    Returns
    -------
    (annotated_frame, emotion_label)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    emotion = "neutral"

    for (x, y, w, h) in faces:
        # Same rule-based logic as emotion_detection.py
        if h > 250:
            emotion = "happy"
        elif h < 120:
            emotion = "sad"
        else:
            emotion = "neutral"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        emotion,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return frame, emotion
