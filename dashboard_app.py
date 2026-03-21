"""
Althera — Real-Time Cognitive Monitoring Dashboard
====================================================
Central controller orchestrating the entire cognitive monitoring pipeline.
Run with:  streamlit run dashboard_app.py
"""

import streamlit as st
import time
import random
import numpy as np
import cv2
import plotly.graph_objects as go
from datetime import datetime

from dashboard_bridge import (
    SensorReader,
    predict_cognitive_score,
    generate_ai_report,
    detect_emotion_from_frame,
)
from dashboard_styles import DARK_CSS, PLOTLY_DARK_TEMPLATE

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Althera — Cognitive Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════
DEFAULTS = {
    "current_step": 1,
    "sensor_reader": None,
    "sensor_data": {"HeartRate": 0, "SpO2": 0, "Motion": 0},
    "sensor_history": [],
    "sensor_connected": False,
    # Reaction test state
    "reaction_results": {},
    "rt_simple_times": [],
    "rt_choice_times": [],
    "rt_choice_correct": 0,
    "rt_taps": 0,
    "rt_phase": "idle",
    "rt_trial": 0,
    "rt_start_time": None,
    "rt_current_color": None,
    "rt_tap_start": None,
    "rt_tap_counts": [],
    # Memory test state
    "mem_phase": "idle",
    "mem_trial": 0,
    "mem_words": [],
    "mem_word_scores": [],
    "mem_numbers": [],
    "mem_number_scores": [],
    "mem_stroop_word": "",
    "mem_stroop_color": "",
    "mem_stroop_times": [],
    "mem_stroop_correct": 0,
    "mem_stroop_trial": 0,
    "mem_show_time": None,
    # Emotion state
    "emotion_results": [],
    "emotion_running": False,
    # Results
    "cognitive_score": None,
    "cognitive_status": None,
    "ai_report": None,
    "all_data": {},
    # Step completion flags
    "step_completed": {i: False for i in range(1, 9)},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def mark_step_done(step):
    st.session_state.step_completed[step] = True
    if st.session_state.current_step == step:
        st.session_state.current_step = step + 1


# ══════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION & PROGRESS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="main-title">🧠 Althera</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Cognitive Health Monitor</p>', unsafe_allow_html=True)
    st.markdown("---")

    STEPS = [
        "Connect Sensor",
        "Collect Sensor Data",
        "Reaction Tests",
        "Memory Tests",
        "Emotion Detection",
        "ML Prediction",
        "AI Interpretation",
        "View Results",
    ]
    st.markdown("#### 📋 Workflow Progress")
    for i, name in enumerate(STEPS, 1):
        if st.session_state.step_completed.get(i):
            st.markdown(f'<div class="step-done">✅ Step {i}: {name}</div>', unsafe_allow_html=True)
        elif i == st.session_state.current_step:
            st.markdown(f'<div class="step-active">▶ Step {i}: {name}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="step-pending">○ Step {i}: {name}</div>', unsafe_allow_html=True)

    st.markdown("---")
    # Manual step jump
    st.session_state.current_step = st.selectbox(
        "Jump to step", range(1, 9),
        index=st.session_state.current_step - 1,
        format_func=lambda x: f"Step {x}: {STEPS[x-1]}",
    )

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🧠 Althera Cognitive Health Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time cognitive monitoring • Alzheimer\'s early indicator detection</p>', unsafe_allow_html=True)
st.markdown("")

step = st.session_state.current_step

# ══════════════════════════════════════════════════════════════
# STEP 1 & 2 — SENSOR CONNECTION & LIVE DATA
# ══════════════════════════════════════════════════════════════
if step in (1, 2):
    st.markdown("## 📡 Heartbeat Sensor Panel")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Live Sensor Stream")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔗 Connect Sensor", use_container_width=True):
                try:
                    reader = SensorReader()
                    reader.start()
                    st.session_state.sensor_reader = reader
                    st.session_state.sensor_connected = True
                    mark_step_done(1)
                    st.success("✅ Sensor connected!")
                except Exception as e:
                    st.error(f"❌ {e}")
        with c2:
            if st.button("📡 Use Simulated Data", use_container_width=True):
                st.session_state.sensor_connected = True
                st.session_state.sensor_data = {
                    "HeartRate": round(random.uniform(65, 85), 1),
                    "SpO2": round(random.uniform(94, 99), 1),
                    "Motion": round(random.uniform(0.5, 3.0), 2),
                }
                # Generate simulated history
                now = time.time()
                hist = []
                for i in range(60):
                    hist.append({
                        "Timestamp": datetime.fromtimestamp(now - 60 + i).strftime("%H:%M:%S"),
                        "HeartRate": round(72 + random.gauss(0, 3), 1),
                        "SpO2": round(96 + random.gauss(0, 1), 1),
                        "Motion": round(1.5 + random.gauss(0, 0.5), 2),
                    })
                st.session_state.sensor_history = hist
                st.session_state.sensor_data = {
                    "HeartRate": hist[-1]["HeartRate"],
                    "SpO2": hist[-1]["SpO2"],
                    "Motion": hist[-1]["Motion"],
                }
                mark_step_done(1)
                mark_step_done(2)
                st.rerun()

        # Live data display
        if st.session_state.sensor_connected:
            reader = st.session_state.sensor_reader
            if reader and reader.connected:
                data = reader.get_latest()
                st.session_state.sensor_data = {
                    "HeartRate": data.get("HeartRate", 0),
                    "SpO2": data.get("SpO2", 0),
                    "Motion": data.get("Motion", 0),
                }
                buf = reader.get_buffer_df()
                if not buf.empty:
                    st.session_state.sensor_history = buf.to_dict("records")
                if len(st.session_state.sensor_history) >= 10:
                    mark_step_done(2)

            sd = st.session_state.sensor_data
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'''<div class="metric-card">
                    <div class="metric-value">{sd["HeartRate"]}</div>
                    <div class="metric-label">❤️ Heart Rate (bpm)</div>
                </div>''', unsafe_allow_html=True)
            with m2:
                st.markdown(f'''<div class="metric-card">
                    <div class="metric-value">{sd["SpO2"]}</div>
                    <div class="metric-label">🫁 SpO2 (%)</div>
                </div>''', unsafe_allow_html=True)
            with m3:
                st.markdown(f'''<div class="metric-card">
                    <div class="metric-value">{sd["Motion"]}</div>
                    <div class="metric-label">🏃 Motion Level</div>
                </div>''', unsafe_allow_html=True)

            # Heart rate chart
            if st.session_state.sensor_history:
                hist = st.session_state.sensor_history
                timestamps = [r.get("Timestamp", "") for r in hist]
                hr_vals = [float(r.get("HeartRate", 0)) for r in hist]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps, y=hr_vals, mode="lines",
                    fill="tozeroy",
                    line=dict(color="#f472b6", width=2),
                    fillcolor="rgba(244,114,182,0.1)",
                    name="Heart Rate",
                ))
                fig.update_layout(
                    title="Heart Rate Over Time",
                    yaxis_title="BPM",
                    height=300,
                    **PLOTLY_DARK_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
        st.markdown("### Connection Status")
        if st.session_state.sensor_connected:
            st.markdown("🟢 **Connected**")
            sd = st.session_state.sensor_data
            st.markdown(f"- Heart Rate: **{sd['HeartRate']} bpm**")
            st.markdown(f"- SpO2: **{sd['SpO2']}%**")
            st.markdown(f"- Motion: **{sd['Motion']}**")
            st.markdown(f"- Samples: **{len(st.session_state.sensor_history)}**")
        else:
            st.markdown("🔴 **Not Connected**")
            st.info("Connect the heartbeat sensor or use simulated data to proceed.")

        if st.session_state.step_completed.get(2):
            if st.button("➡️ Proceed to Tests", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STEP 3 — REACTION TESTS
# ══════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown("## ⚡ Reaction Tests")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Simple Reaction", "🔴🟢 Choice Reaction", "👆 Finger Tapping"])

    # ── Simple Reaction Test ──
    with tab1:
        st.markdown("**Press the button as fast as you can when you see GO!**")
        phase = st.session_state.rt_phase

        if phase == "idle" or phase not in ("simple_wait", "simple_go", "simple_done"):
            if st.button("▶ Start Simple Reaction Test", key="start_simple"):
                st.session_state.rt_simple_times = []
                st.session_state.rt_trial = 0
                st.session_state.rt_phase = "simple_wait"
                st.session_state.rt_start_time = time.time() + random.uniform(2, 4)
                st.rerun()

        if phase == "simple_wait":
            elapsed = time.time() - (st.session_state.rt_start_time - random.uniform(2, 4))
            if time.time() >= st.session_state.rt_start_time:
                st.session_state.rt_phase = "simple_go"
                st.session_state.rt_start_time = time.time()
                st.rerun()
            else:
                st.markdown(f'<div class="test-display">⏳ Wait... (Trial {st.session_state.rt_trial + 1}/3)</div>', unsafe_allow_html=True)
                time.sleep(0.5)
                st.rerun()

        if phase == "simple_go":
            st.markdown('<div class="test-display" style="color:#4ade80; border-color:rgba(74,222,128,0.3);">🟢 GO! Click NOW!</div>', unsafe_allow_html=True)
            if st.button("⚡ CLICK!", key="simple_click", use_container_width=True):
                rt = round((time.time() - st.session_state.rt_start_time) * 1000, 2)
                st.session_state.rt_simple_times.append(rt)
                st.session_state.rt_trial += 1
                if st.session_state.rt_trial >= 3:
                    st.session_state.rt_phase = "simple_done"
                else:
                    st.session_state.rt_phase = "simple_wait"
                    st.session_state.rt_start_time = time.time() + random.uniform(2, 4)
                st.rerun()

        if phase == "simple_done":
            times = st.session_state.rt_simple_times
            avg = round(sum(times) / len(times), 2) if times else 0
            st.success(f"✅ Average Simple Reaction Time: **{avg} ms**")
            for i, t in enumerate(times):
                st.markdown(f"  Trial {i+1}: {t} ms")
            st.session_state.reaction_results["simple_reaction_ms"] = avg

    # ── Choice Reaction Test ──
    with tab2:
        st.markdown("**Click the button matching the displayed color (R for Red, G for Green).**")
        phase = st.session_state.rt_phase

        if phase not in ("choice_show", "choice_done"):
            if st.button("▶ Start Choice Reaction Test", key="start_choice"):
                st.session_state.rt_choice_times = []
                st.session_state.rt_choice_correct = 0
                st.session_state.rt_trial = 0
                st.session_state.rt_current_color = random.choice(["RED", "GREEN"])
                st.session_state.rt_start_time = time.time()
                st.session_state.rt_phase = "choice_show"
                st.rerun()

        if phase == "choice_show":
            color = st.session_state.rt_current_color
            hex_c = "#f87171" if color == "RED" else "#4ade80"
            st.markdown(f'<div class="test-display" style="color:{hex_c};">{color} (Trial {st.session_state.rt_trial + 1}/3)</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔴 RED", key="choice_red", use_container_width=True):
                    rt = round((time.time() - st.session_state.rt_start_time) * 1000, 2)
                    st.session_state.rt_choice_times.append(rt)
                    if color == "RED":
                        st.session_state.rt_choice_correct += 1
                    st.session_state.rt_trial += 1
                    if st.session_state.rt_trial >= 3:
                        st.session_state.rt_phase = "choice_done"
                    else:
                        st.session_state.rt_current_color = random.choice(["RED", "GREEN"])
                        st.session_state.rt_start_time = time.time()
                    st.rerun()
            with c2:
                if st.button("🟢 GREEN", key="choice_green", use_container_width=True):
                    rt = round((time.time() - st.session_state.rt_start_time) * 1000, 2)
                    st.session_state.rt_choice_times.append(rt)
                    if color == "GREEN":
                        st.session_state.rt_choice_correct += 1
                    st.session_state.rt_trial += 1
                    if st.session_state.rt_trial >= 3:
                        st.session_state.rt_phase = "choice_done"
                    else:
                        st.session_state.rt_current_color = random.choice(["RED", "GREEN"])
                        st.session_state.rt_start_time = time.time()
                    st.rerun()

        if phase == "choice_done":
            times = st.session_state.rt_choice_times
            avg = round(sum(times) / len(times), 2) if times else 0
            acc = round((st.session_state.rt_choice_correct / 3) * 100, 2)
            st.success(f"✅ Avg Choice RT: **{avg} ms** | Accuracy: **{acc}%**")
            st.session_state.reaction_results["choice_reaction_ms"] = avg
            st.session_state.reaction_results["choice_accuracy_percent"] = acc

    # ── Finger Tapping Test ──
    with tab3:
        st.markdown("**Tap the button as many times as you can in 5 seconds!**")
        phase = st.session_state.rt_phase

        if phase not in ("tap_active", "tap_done"):
            if st.button("▶ Start Finger Tapping Test", key="start_tap"):
                st.session_state.rt_taps = 0
                st.session_state.rt_tap_start = time.time()
                st.session_state.rt_trial = 0
                st.session_state.rt_tap_counts = []
                st.session_state.rt_phase = "tap_active"
                st.rerun()

        if phase == "tap_active":
            elapsed = time.time() - st.session_state.rt_tap_start
            remaining = max(0, 5.0 - elapsed)
            st.markdown(f'<div class="test-display">⏱️ {remaining:.1f}s left | Taps: {st.session_state.rt_taps} (Trial {st.session_state.rt_trial + 1}/3)</div>', unsafe_allow_html=True)

            if remaining > 0:
                if st.button("👆 TAP!", key="tap_btn", use_container_width=True):
                    st.session_state.rt_taps += 1
                    st.rerun()
            else:
                st.session_state.rt_tap_counts.append(st.session_state.rt_taps)
                st.session_state.rt_trial += 1
                if st.session_state.rt_trial >= 3:
                    st.session_state.rt_phase = "tap_done"
                else:
                    st.session_state.rt_taps = 0
                    st.session_state.rt_tap_start = time.time()
                st.rerun()

        if phase == "tap_done":
            counts = st.session_state.rt_tap_counts
            avg = round(sum(counts) / len(counts), 2) if counts else 0
            st.success(f"✅ Avg Taps per 5s: **{avg}**")
            for i, c in enumerate(counts):
                st.markdown(f"  Trial {i+1}: {c} taps")
            st.session_state.reaction_results["finger_taps"] = avg

    st.markdown('</div>', unsafe_allow_html=True)

    # Summary & proceed
    if st.session_state.reaction_results:
        st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
        st.markdown("### 📊 Reaction Test Results")
        for k, v in st.session_state.reaction_results.items():
            st.markdown(f"- **{k}**: {v}")
        st.markdown('</div>', unsafe_allow_html=True)

    all_done = all(k in st.session_state.reaction_results for k in ["simple_reaction_ms", "choice_reaction_ms", "finger_taps"])
    if all_done:
        mark_step_done(3)
        if st.button("➡️ Proceed to Memory Tests", use_container_width=True):
            st.session_state.rt_phase = "idle"
            st.session_state.current_step = 4
            st.rerun()
    elif st.session_state.reaction_results:
        st.info("Complete all 3 reaction tests to proceed, or skip ahead.")
        if st.button("⏭️ Skip to Memory Tests", use_container_width=True):
            for k in ["simple_reaction_ms", "choice_reaction_ms", "finger_taps"]:
                st.session_state.reaction_results.setdefault(k, 0)
            mark_step_done(3)
            st.session_state.current_step = 4
            st.rerun()

# ══════════════════════════════════════════════════════════════
# STEP 4 — MEMORY TESTS
# ══════════════════════════════════════════════════════════════
elif step == 4:
    st.markdown("## 🧩 Memory & Cognitive Tests")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Word Recall", "🔢 Number Recall", "🎨 Stroop Test"])

    WORD_BANK = ["apple", "chair", "river", "house", "book", "green", "clock", "phone"]
    COLORS = ["RED", "GREEN", "BLUE", "YELLOW"]

    # ── Word Recall ──
    with tab1:
        st.markdown("**Memorize the words, then recall as many as you can.**")
        phase = st.session_state.mem_phase

        if phase not in ("word_show", "word_recall", "word_done"):
            if st.button("▶ Start Word Recall", key="start_word"):
                st.session_state.mem_word_scores = []
                st.session_state.mem_trial = 0
                words = random.sample(WORD_BANK, 5)
                st.session_state.mem_words = words
                st.session_state.mem_show_time = time.time()
                st.session_state.mem_phase = "word_show"
                st.rerun()

        if phase == "word_show":
            elapsed = time.time() - st.session_state.mem_show_time
            if elapsed < 6:
                words_display = " • ".join(st.session_state.mem_words)
                st.markdown(f'<div class="test-display" style="font-size:1.8rem;">{words_display}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="test-instruction">Memorize these words! ({6-elapsed:.0f}s remaining)</div>', unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.mem_phase = "word_recall"
                st.rerun()

        if phase == "word_recall":
            st.markdown(f'<div class="test-instruction">Trial {st.session_state.mem_trial + 1}/3 — Type the words you remember:</div>', unsafe_allow_html=True)
            recall = st.text_input("Enter recalled words (space-separated):", key=f"word_recall_{st.session_state.mem_trial}")
            if st.button("Submit", key=f"word_submit_{st.session_state.mem_trial}"):
                recalled = recall.lower().split()
                correct = len(set(recalled) & set(st.session_state.mem_words))
                st.session_state.mem_word_scores.append(correct)
                st.session_state.mem_trial += 1
                if st.session_state.mem_trial >= 3:
                    st.session_state.mem_phase = "word_done"
                else:
                    words = random.sample(WORD_BANK, 5)
                    st.session_state.mem_words = words
                    st.session_state.mem_show_time = time.time()
                    st.session_state.mem_phase = "word_show"
                st.rerun()

        if phase == "word_done":
            scores = st.session_state.mem_word_scores
            avg = round(sum(scores) / len(scores), 2) if scores else 0
            st.success(f"✅ Average Word Recall: **{avg}/5**")
            st.session_state.reaction_results["word_recall_score"] = avg

    # ── Number Recall ──
    with tab2:
        st.markdown("**Memorize the number sequence, then enter it back.**")
        phase = st.session_state.mem_phase

        if phase not in ("num_show", "num_recall", "num_done"):
            if st.button("▶ Start Number Recall", key="start_num"):
                st.session_state.mem_number_scores = []
                st.session_state.mem_trial = 0
                nums = [random.randint(0, 9) for _ in range(6)]
                st.session_state.mem_numbers = nums
                st.session_state.mem_show_time = time.time()
                st.session_state.mem_phase = "num_show"
                st.rerun()

        if phase == "num_show":
            elapsed = time.time() - st.session_state.mem_show_time
            if elapsed < 5:
                nums_display = " ".join(str(n) for n in st.session_state.mem_numbers)
                st.markdown(f'<div class="test-display">{nums_display}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="test-instruction">Memorize! ({5-elapsed:.0f}s remaining)</div>', unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.mem_phase = "num_recall"
                st.rerun()

        if phase == "num_recall":
            st.markdown(f'<div class="test-instruction">Trial {st.session_state.mem_trial + 1}/3 — Enter the numbers:</div>', unsafe_allow_html=True)
            recall = st.text_input("Enter numbers (space-separated):", key=f"num_recall_{st.session_state.mem_trial}")
            if st.button("Submit", key=f"num_submit_{st.session_state.mem_trial}"):
                response = recall.split()
                seq = st.session_state.mem_numbers
                correct = sum(1 for j in range(min(len(seq), len(response))) if response[j] == str(seq[j]))
                st.session_state.mem_number_scores.append(correct)
                st.session_state.mem_trial += 1
                if st.session_state.mem_trial >= 3:
                    st.session_state.mem_phase = "num_done"
                else:
                    nums = [random.randint(0, 9) for _ in range(6)]
                    st.session_state.mem_numbers = nums
                    st.session_state.mem_show_time = time.time()
                    st.session_state.mem_phase = "num_show"
                st.rerun()

        if phase == "num_done":
            scores = st.session_state.mem_number_scores
            avg = round(sum(scores) / len(scores), 2) if scores else 0
            st.success(f"✅ Average Number Recall: **{avg}/6**")
            st.session_state.reaction_results["number_recall_score"] = avg

    # ── Stroop Test ──
    with tab3:
        st.markdown("**Type the COLOR of the text, not the word itself.**")
        phase = st.session_state.mem_phase

        if phase not in ("stroop_show", "stroop_done"):
            if st.button("▶ Start Stroop Test", key="start_stroop"):
                st.session_state.mem_stroop_times = []
                st.session_state.mem_stroop_correct = 0
                st.session_state.mem_stroop_trial = 0
                word = random.choice(COLORS)
                color = random.choice(COLORS)
                st.session_state.mem_stroop_word = word
                st.session_state.mem_stroop_color = color
                st.session_state.rt_start_time = time.time()
                st.session_state.mem_phase = "stroop_show"
                st.rerun()

        if phase == "stroop_show":
            word = st.session_state.mem_stroop_word
            color = st.session_state.mem_stroop_color
            color_map = {"RED": "#f87171", "GREEN": "#4ade80", "BLUE": "#60a5fa", "YELLOW": "#fbbf24"}
            hex_c = color_map.get(color, "#e0e0f0")
            st.markdown(f'<div class="test-display" style="color:{hex_c}; font-size:4rem;">{word}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="test-instruction">Trial {st.session_state.mem_stroop_trial + 1}/3 — What COLOR is this text?</div>', unsafe_allow_html=True)

            response = st.text_input("Enter the COLOR:", key=f"stroop_{st.session_state.mem_stroop_trial}")
            if st.button("Submit", key=f"stroop_submit_{st.session_state.mem_stroop_trial}"):
                rt = round((time.time() - st.session_state.rt_start_time) * 1000, 2)
                st.session_state.mem_stroop_times.append(rt)
                if response.upper() == color:
                    st.session_state.mem_stroop_correct += 1
                st.session_state.mem_stroop_trial += 1
                if st.session_state.mem_stroop_trial >= 3:
                    st.session_state.mem_phase = "stroop_done"
                else:
                    word = random.choice(COLORS)
                    color = random.choice(COLORS)
                    st.session_state.mem_stroop_word = word
                    st.session_state.mem_stroop_color = color
                    st.session_state.rt_start_time = time.time()
                st.rerun()

        if phase == "stroop_done":
            times = st.session_state.mem_stroop_times
            avg_rt = round(sum(times) / len(times), 2) if times else 0
            acc = round((st.session_state.mem_stroop_correct / 3) * 100, 2)
            st.success(f"✅ Avg Stroop RT: **{avg_rt} ms** | Accuracy: **{acc}%**")
            st.session_state.reaction_results["stroop_accuracy_percent"] = acc

    st.markdown('</div>', unsafe_allow_html=True)

    mem_keys = ["word_recall_score", "number_recall_score", "stroop_accuracy_percent"]
    all_mem_done = all(k in st.session_state.reaction_results for k in mem_keys)
    if all_mem_done:
        mark_step_done(4)
        if st.button("➡️ Proceed to Emotion Detection", use_container_width=True):
            st.session_state.mem_phase = "idle"
            st.session_state.current_step = 5
            st.rerun()
    else:
        st.info("Complete all 3 memory tests to proceed, or skip ahead.")
        if st.button("⏭️ Skip to Emotion Detection", use_container_width=True):
            for k in mem_keys:
                st.session_state.reaction_results.setdefault(k, 0)
            mark_step_done(4)
            st.session_state.current_step = 5
            st.rerun()

# ══════════════════════════════════════════════════════════════
# STEP 5 — EMOTION DETECTION
# ══════════════════════════════════════════════════════════════
elif step == 5:
    st.markdown("## 😊 Facial Emotion Detection")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("Capture your facial expression using the webcam. The system uses face detection to analyze your emotional state.")

    cam_input = st.camera_input("📸 Take a photo for emotion analysis")

    if cam_input is not None:
        file_bytes = np.frombuffer(cam_input.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, emotion = detect_emotion_from_frame(frame)
        st.session_state.emotion_results.append(emotion)

        col1, col2 = st.columns([2, 1])
        with col1:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Analyzed Frame", use_container_width=True)
        with col2:
            badge_class = f"emotion-{emotion}"
            st.markdown(f'<span class="emotion-badge {badge_class}">{emotion}</span>', unsafe_allow_html=True)
            st.markdown(f"**Samples collected:** {len(st.session_state.emotion_results)}")

            if st.session_state.emotion_results:
                from collections import Counter
                counts = Counter(st.session_state.emotion_results)
                fig = go.Figure(data=[go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.5,
                    marker=dict(colors=["#4ade80", "#60a5fa", "#fbbf24"]),
                )])
                fig.update_layout(height=250, **PLOTLY_DARK_TEMPLATE, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

    if len(st.session_state.emotion_results) >= 1:
        mark_step_done(5)
        if st.button("➡️ Run AI Prediction", use_container_width=True):
            st.session_state.current_step = 6
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STEP 6 — ML PREDICTION
# ══════════════════════════════════════════════════════════════
elif step == 6:
    st.markdown("## 🤖 ML Cognitive Score Prediction")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Compile all data
    sd = st.session_state.sensor_data
    rr = st.session_state.reaction_results
    all_data = {
        "HeartRate": sd.get("HeartRate", 72),
        "SpO2": sd.get("SpO2", 96),
        "Motion": sd.get("Motion", 1.0),
        "simple_reaction_ms": rr.get("simple_reaction_ms", 0),
        "choice_reaction_ms": rr.get("choice_reaction_ms", 0),
        "finger_taps": rr.get("finger_taps", 0),
        "word_recall_score": rr.get("word_recall_score", 0),
        "number_recall_score": rr.get("number_recall_score", 0),
        "stroop_accuracy_percent": rr.get("stroop_accuracy_percent", 0),
    }

    st.markdown("### 📋 Collected Data Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sensor Data**")
        st.markdown(f"- Heart Rate: `{all_data['HeartRate']} bpm`")
        st.markdown(f"- SpO2: `{all_data['SpO2']}%`")
        st.markdown(f"- Motion: `{all_data['Motion']}`")
    with c2:
        st.markdown("**Reaction Tests**")
        st.markdown(f"- Simple RT: `{all_data['simple_reaction_ms']} ms`")
        st.markdown(f"- Choice RT: `{all_data['choice_reaction_ms']} ms`")
        st.markdown(f"- Finger Taps: `{all_data['finger_taps']}`")
    with c3:
        st.markdown("**Memory Tests**")
        st.markdown(f"- Word Recall: `{all_data['word_recall_score']}`")
        st.markdown(f"- Number Recall: `{all_data['number_recall_score']}`")
        st.markdown(f"- Stroop Accuracy: `{all_data['stroop_accuracy_percent']}%`")

    if st.button("🧠 Run ML Prediction", use_container_width=True):
        with st.spinner("Running cognitive model..."):
            try:
                score, status = predict_cognitive_score(all_data)
                st.session_state.cognitive_score = score
                st.session_state.cognitive_status = status
                st.session_state.all_data = all_data
                st.session_state.all_data["cognitive_score"] = score
                st.session_state.all_data["cognitive_status"] = status
                mark_step_done(6)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    if st.session_state.cognitive_score is not None:
        score = st.session_state.cognitive_score
        status = st.session_state.cognitive_status
        if score > 85:
            css_class = "score-excellent"
        elif score > 70:
            css_class = "score-normal"
        elif score > 50:
            css_class = "score-mild"
        else:
            css_class = "score-low"

        st.markdown(f'<div class="score-display {css_class}">{score}</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:1.2rem;'>{status}</p>", unsafe_allow_html=True)

        if st.button("➡️ Generate AI Report", use_container_width=True):
            st.session_state.current_step = 7
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STEP 7 — AI INTERPRETATION
# ══════════════════════════════════════════════════════════════
elif step == 7:
    st.markdown("## 🧬 Generative AI Interpretation")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if st.session_state.ai_report:
        st.markdown(f'<div class="ai-report">{st.session_state.ai_report}</div>', unsafe_allow_html=True)
        mark_step_done(7)
        if st.button("➡️ View Final Results", use_container_width=True):
            st.session_state.current_step = 8
            st.rerun()
    else:
        st.info("This will send your collected data to Ollama (llama3.2) for a detailed cognitive health analysis.")
        if st.button("🧬 Generate AI Report", use_container_width=True):
            with st.spinner("🧠 AI is analyzing your cognitive data... (this may take a minute)"):
                report = generate_ai_report(st.session_state.all_data or st.session_state.sensor_data)
                st.session_state.ai_report = report
                mark_step_done(7)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# STEP 8 — FINAL RESULTS
# ══════════════════════════════════════════════════════════════
elif step == 8:
    st.markdown("## 📊 Final Cognitive Health Report")
    mark_step_done(8)

    # Score display
    score = st.session_state.cognitive_score or 0
    status = st.session_state.cognitive_status or "Not yet predicted"
    if score > 85:
        css_class, color = "score-excellent", "#4ade80"
    elif score > 70:
        css_class, color = "score-normal", "#60a5fa"
    elif score > 50:
        css_class, color = "score-mild", "#fbbf24"
    else:
        css_class, color = "score-low", "#f87171"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
        st.markdown("### Cognitive Score")
        st.markdown(f'<div class="score-display {css_class}">{score}</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:{color};font-weight:600;'>{status}</p>", unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#5a5a7a"),
                bar=dict(color=color),
                bgcolor="rgba(15,15,40,0.3)",
                steps=[
                    dict(range=[0, 50], color="rgba(248,113,113,0.15)"),
                    dict(range=[50, 70], color="rgba(251,191,36,0.15)"),
                    dict(range=[70, 85], color="rgba(96,165,250,0.15)"),
                    dict(range=[85, 100], color="rgba(74,222,128,0.15)"),
                ],
            ),
        ))
        fig.update_layout(height=250, **PLOTLY_DARK_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Sensor summary
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📡 Sensor Summary")
        sd = st.session_state.sensor_data
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-value">{sd.get("HeartRate", 0)}</div>
                <div class="metric-label">❤️ Heart Rate</div>
            </div>''', unsafe_allow_html=True)
        with m2:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-value">{sd.get("SpO2", 0)}</div>
                <div class="metric-label">🫁 SpO2</div>
            </div>''', unsafe_allow_html=True)
        with m3:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-value">{sd.get("Motion", 0)}</div>
                <div class="metric-label">🏃 Motion</div>
            </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Test performance
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Test Performance")
        rr = st.session_state.reaction_results
        features = {
            "Simple RT (ms)": rr.get("simple_reaction_ms", 0),
            "Choice RT (ms)": rr.get("choice_reaction_ms", 0),
            "Finger Taps": rr.get("finger_taps", 0),
            "Word Recall": rr.get("word_recall_score", 0),
            "Number Recall": rr.get("number_recall_score", 0),
            "Stroop Acc %": rr.get("stroop_accuracy_percent", 0),
        }
        fig = go.Figure(data=[go.Bar(
            x=list(features.keys()),
            y=list(features.values()),
            marker=dict(
                color=["#f472b6", "#a78bfa", "#60a5fa", "#4ade80", "#fbbf24", "#f87171"],
                line=dict(width=0),
            ),
        )])
        fig.update_layout(title="Test Metrics", height=280, **PLOTLY_DARK_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # AI Report
    if st.session_state.ai_report:
        st.markdown('<div class="glass-card-accent">', unsafe_allow_html=True)
        st.markdown("### 🧬 AI Cognitive Analysis")
        st.markdown(f'<div class="ai-report">{st.session_state.ai_report}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Emotion summary
    if st.session_state.emotion_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 😊 Emotion Analysis")
        from collections import Counter
        counts = Counter(st.session_state.emotion_results)
        dominant = counts.most_common(1)[0][0]
        st.markdown(f"Dominant emotion: **{dominant.upper()}** ({counts[dominant]}/{len(st.session_state.emotion_results)} samples)")
        st.markdown('</div>', unsafe_allow_html=True)

    # Restart
    st.markdown("---")
    if st.button("🔄 Start New Assessment", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#4a4a6a; font-size:0.8rem;">'
    '🧠 Althera Cognitive Health Monitor • Early Alzheimer\'s Indicator Detection System'
    '</p>',
    unsafe_allow_html=True,
)
