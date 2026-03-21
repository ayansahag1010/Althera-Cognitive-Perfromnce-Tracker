"""
Althera Dashboard — Helpers & Constants
Data loading, theme, and utility functions.
"""
import os, random, pandas as pd, numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Word banks for tests ─────────────────────────────────────────────
WORD_BANK = ["apple","chair","river","house","book","green","clock","phone",
             "music","train","cloud","stone","flame","ocean","bread","light"]
STROOP_COLORS = ["RED","GREEN","BLUE","YELLOW"]
STROOP_HEX = {"RED":"#ef4444","GREEN":"#22c55e","BLUE":"#3b82f6","YELLOW":"#eab308"}

# ── Theme ─────────────────────────────────────────────────────────────
C = {
    "bg":"#0a0e17","card":"rgba(15,23,42,0.85)","card_border":"rgba(99,102,241,0.25)",
    "accent":"#6366f1","accent_light":"#818cf8","text":"#e2e8f0","muted":"#94a3b8",
    "success":"#22c55e","warning":"#f59e0b","danger":"#ef4444","info":"#06b6d4",
    "heart":"#f43f5e","spo2":"#3b82f6","motion":"#a855f7",
}
CARD = {
    "backgroundColor":C["card"],"borderRadius":"16px",
    "border":f"1px solid {C['card_border']}","padding":"24px",
    "backdropFilter":"blur(12px)","boxShadow":"0 8px 32px rgba(0,0,0,0.3)",
}
PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,system-ui,sans-serif", color=C["text"], size=12),
    margin=dict(l=50,r=30,t=40,b=40),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

# ── Data helpers ──────────────────────────────────────────────────────
def _csv(name, cols):
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, name))
        return df if not df.empty else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)

def load_sensor():
    cols=["Timestamp","AX","AY","AZ","Motion","HeartRate","SpO2"]
    df=_csv("sensor_data.csv",cols)
    for c in cols[1:]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["Timestamp"]=pd.to_datetime(df["Timestamp"],errors="coerce")
    df.dropna(subset=["Timestamp"],inplace=True)
    return df

def load_reaction():
    cols=["timestamp","simple_reaction_ms","choice_reaction_ms","choice_accuracy_percent","finger_taps"]
    df=_csv("reaction_results.csv",cols)
    for c in cols[1:]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def load_memory():
    cols=["timestamp","word_recall_score","number_recall_score","stroop_reaction_ms","stroop_accuracy_percent"]
    df=_csv("memory_results.csv",cols)
    for c in cols[1:]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def load_combined():
    cols=["HeartRate","SpO2","Motion","simple_reaction_ms","choice_reaction_ms",
          "finger_taps","word_recall_score","number_recall_score","stroop_accuracy_percent"]
    df=_csv("combined_data.csv",cols)
    for c in cols:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    return df

def load_ai_report():
    try:
        with open(os.path.join(BASE_DIR,"ai_report.txt"),"r",encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "No AI report available."

def cognitive_score(cd):
    if cd.empty: return 0.0
    r=cd.iloc[-1]
    s=100-(r.get("simple_reaction_ms",0)*0.05)+(r.get("word_recall_score",0)*5)+(r.get("stroop_accuracy_percent",0)*0.2)
    return round(max(0,min(100,s)),1)

def save_sensor_row(hr, spo2, motion, ax=0, ay=0, az=0):
    """Append a row to sensor_data.csv."""
    import csv
    path = os.path.join(BASE_DIR, "sensor_data.csv")
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Timestamp","AX","AY","AZ","Motion","HeartRate","SpO2"])
        w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ax, ay, az, motion, hr, spo2])

def save_test_results(rt_avg=None, mem_avg=None, stroop_rt=None, stroop_acc=None, taps=None):
    """Save test results to reaction_results.csv and memory_results.csv."""
    import csv
    # Reaction
    rpath = os.path.join(BASE_DIR, "reaction_results.csv")
    exists = os.path.isfile(rpath) and os.path.getsize(rpath) > 0
    with open(rpath, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","simple_reaction_ms","choice_reaction_ms","choice_accuracy_percent","finger_taps"])
        w.writerow([datetime.now(), rt_avg or "", "", "", taps or ""])
    # Memory
    mpath = os.path.join(BASE_DIR, "memory_results.csv")
    exists = os.path.isfile(mpath) and os.path.getsize(mpath) > 0
    with open(mpath, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","word_recall_score","number_recall_score","stroop_reaction_ms","stroop_accuracy_percent"])
        w.writerow([datetime.now(), mem_avg or "", "", stroop_rt or "", stroop_acc or ""])

def merge_and_save():
    """Merge latest sensor + test data into combined_data.csv."""
    s = load_sensor(); r = load_reaction(); m = load_memory()
    ls = s.tail(1).reset_index(drop=True) if not s.empty else pd.DataFrame()
    lr = r.tail(1).reset_index(drop=True) if not r.empty else pd.DataFrame()
    lm = m.tail(1).reset_index(drop=True) if not m.empty else pd.DataFrame()
    combined = pd.concat([ls, lr, lm], axis=1).fillna(0)
    combined.to_csv(os.path.join(BASE_DIR, "combined_data.csv"), index=False)
    return combined

def run_ai_analysis(sensor_d, test_d, emotion_d):
    """Try Ollama, fallback to heuristic analysis."""
    hr=sensor_d.get("hr",72); spo2=sensor_d.get("spo2",96); motion=sensor_d.get("motion",1.0)
    rt=test_d.get("rt_avg",0); mem=test_d.get("mem_avg",0)
    stroop_acc=test_d.get("stroop_acc",0); taps=test_d.get("tap_avg",0)
    emotion=emotion_d.get("emotion","neutral")

    prompt = f"""Analyze this cognitive health data for early Alzheimer's indicators:
Heart Rate: {hr} bpm | SpO2: {spo2}% | Motion: {motion}
Reaction Time: {rt:.0f} ms | Word Recall: {mem:.1f}/5 | Stroop Accuracy: {stroop_acc:.0f}%
Finger Taps: {taps:.0f} | Emotional State: {emotion}

Provide: 1) Cognitive health summary 2) Risk indicators 3) Recommendations"""

    try:
        import ollama
        resp = ollama.chat(model="llama3.2", messages=[{"role":"user","content":prompt}])
        report = resp["message"]["content"]
    except Exception:
        # Heuristic fallback
        score = 100 - (rt*0.05) + (mem*5) + (stroop_acc*0.2)
        score = max(0, min(100, score))
        if score > 85: level, risk = "Excellent", "Low"
        elif score > 70: level, risk = "Normal", "Low-Moderate"
        elif score > 50: level, risk = "Mild Concern", "Moderate"
        else: level, risk = "Needs Attention", "Elevated"

        report = f"""**Cognitive Health Summary**
* Overall Score: {score:.1f}/100 — {level}
* Heart Rate: {hr} bpm {'(normal)' if 60<=hr<=100 else '(abnormal)'}
* SpO2: {spo2}% {'(normal)' if spo2>=95 else '(low — monitor closely)'}
* Reaction Time: {rt:.0f} ms {'(good)' if rt<400 else '(elevated — may indicate slower processing)'}
* Word Recall: {mem:.1f}/5 {'(strong)' if mem>=4 else '(needs improvement)'}
* Emotional State: {emotion}

**Risk Assessment**
* Alzheimer's Risk Level: {risk}
* {'No significant concerns detected.' if score>70 else 'Some cognitive indicators warrant monitoring.'}

**Recommendations**
* {'Continue regular cognitive exercises.' if score>70 else 'Consider scheduling a professional cognitive assessment.'}
* Regular physical activity and adequate sleep are recommended.
* Practice memory exercises such as puzzles and word games.
* Monitor SpO2 levels regularly.
* Maintain social engagement and mental stimulation."""

    with open(os.path.join(BASE_DIR, "ai_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report
