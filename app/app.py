import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Predictive Maintenance System", page_icon="🛠️", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "models")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "config.json") as f:
        config = json.load(f)
    return model, config


model, config = load_artifacts()
THRESHOLD = config["threshold"]
RANGES = config["ranges"]

INT_COLS = ["Operational_Hours", "Last_Maintenance_Days_Ago", "Maintenance_History_Count",
            "Failure_History_Count", "Error_Codes_Last_30_Days", "Machine_Age"]

DEFAULTS = {
    "Operational_Hours": 50000, "Temperature_C": 60.0, "Vibration_mms": 10.0,
    "Sound_dB": 75.0, "Oil_Level_pct": 70.0, "Coolant_Level_pct": 65.0,
    "Power_Consumption_kW": 150.0, "Last_Maintenance_Days_Ago": 180,
    "Maintenance_History_Count": 5, "Failure_History_Count": 2,
    "Error_Codes_Last_30_Days": 3, "Machine_Age": 13,
}
HELP = {
    "Temperature_C": "Machine ka operating temperature. Zyada garam hona overheating ka sanket hai.",
    "Vibration_mms": "Machine kitna hil rahi hai. Zyada vibration aksar wear ya misalignment dikhata hai.",
    "Sound_dB": "Machine ki awaaz ka level.",
    "Oil_Level_pct": "Lubrication oil kitna bacha hai.",
    "Coolant_Level_pct": "Coolant kitna bacha hai.",
    "Power_Consumption_kW": "Machine kitni bijli kheench rahi hai.",
    "Operational_Hours": "Machine ab tak kitne ghante chal chuki hai.",
    "Machine_Age": "Machine kitne saal purani hai.",
    "Last_Maintenance_Days_Ago": "Pichhli maintenance ko kitne din ho gaye.",
    "Maintenance_History_Count": "Ab tak kitni baar maintenance hui.",
    "Failure_History_Count": "Pehle kitni baar ye machine fail ho chuki hai.",
    "Error_Codes_Last_30_Days": "Pichhle 30 din mein kitne error codes aaye.",
}

# Ready-made examples so a first-time user can try the app without knowing the values
PRESETS = {
    "Custom (khud daalo)": {},
    "Healthy young machine": {"Operational_Hours": 20000, "Temperature_C": 60.0, "Vibration_mms": 10.0},
    "Old machine, sensors normal": {"Operational_Hours": 95000, "Temperature_C": 60.0, "Vibration_mms": 10.0},
    "Old machine, hot + high vibration": {"Operational_Hours": 95000, "Temperature_C": 75.0, "Vibration_mms": 18.0},
}


def risk_level(prob):
    if prob >= 0.5:
        return "HIGH", "Schedule maintenance now."
    if prob >= THRESHOLD:
        return "MEDIUM", "Inspect this machine soon."
    return "LOW", "Normal operation."


def clamp(value, lo, hi):
    """Keep a default value inside the slider range so Streamlit never errors."""
    return min(max(value, lo), hi)


# ---------- Header ----------
st.title("🛠️ Predictive Maintenance and Failure Forecasting System")
st.caption("Predicts whether a machine is likely to fail within 7 days from its sensor and usage data.")

tab_predict, tab_info = st.tabs(["Prediction", "Model information"])

# ---------- Sidebar inputs ----------
st.sidebar.header("Machine inputs")

example = st.sidebar.selectbox(
    "Quick example", list(PRESETS),
    help="Ek ready example chuno, sliders apne aap set ho jayenge. Phir kisi bhi slider ko badal sakte ho.",
)
DEFAULTS = {**DEFAULTS, **PRESETS[example]}

machine = {"Machine_Type": st.sidebar.selectbox("Machine type", config["machine_types"])}

st.sidebar.subheader("Sensor readings")
for col, label in [("Temperature_C", "Temperature (°C)"),
                   ("Vibration_mms", "Vibration (mm/s)"),
                   ("Sound_dB", "Sound (dB)"),
                   ("Oil_Level_pct", "Oil level (%)"),
                   ("Coolant_Level_pct", "Coolant level (%)"),
                   ("Power_Consumption_kW", "Power consumption (kW)")]:
    lo, hi = RANGES[col]
    machine[col] = st.sidebar.slider(label, float(lo), float(hi),
                                     float(clamp(DEFAULTS[col], lo, hi)), help=HELP[col])

st.sidebar.subheader("Usage and maintenance")
for col, label in [("Operational_Hours", "Operational hours"),
                   ("Machine_Age", "Machine age (years)"),
                   ("Last_Maintenance_Days_Ago", "Days since last maintenance"),
                   ("Maintenance_History_Count", "Past maintenance count"),
                   ("Failure_History_Count", "Past failure count"),
                   ("Error_Codes_Last_30_Days", "Error codes (last 30 days)")]:
    lo, hi = RANGES[col]
    machine[col] = st.sidebar.slider(label, int(lo), int(hi),
                                     int(clamp(DEFAULTS[col], lo, hi)), help=HELP[col])

ai = st.sidebar.selectbox("AI supervision", ["No", "Yes"],
                          help="Kya machine par AI-based monitoring lagi hai.")
machine["AI_Supervision"] = 1 if ai == "Yes" else 0

row = pd.DataFrame([machine])[config["features"]]

# ---------- Prediction tab ----------
with tab_predict:
    st.info(
        "**Kaise use karein:**\n\n"
        "1. Left sidebar mein machine type chuno aur sensor values set karo "
        "(ya 'Quick example' chuno).\n"
        "2. Yahan 7 din mein failure ki probability aur risk level dikhega.\n"
        "3. Neeche 'What-if analysis' mein ek value badal ke dekho ki risk kaise badalta hai."
    )

    prob = float(model.predict_proba(row)[0, 1])
    level, advice = risk_level(prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability (7 days)", f"{prob:.1%}")
    c2.metric("Risk level", level)
    c3.metric("Alert threshold", f"{THRESHOLD:.0%}")
    st.progress(min(prob, 1.0))

    st.caption(
        f"🟢 LOW: probability {THRESHOLD:.0%} se kam, normal operation.  "
        f"🟡 MEDIUM: {THRESHOLD:.0%} se 50% ke beech, jaldi inspection.  "
        f"🔴 HIGH: 50% se zyada, turant maintenance."
    )

    if level == "HIGH":
        st.error(f"HIGH risk: {advice}")
    elif level == "MEDIUM":
        st.warning(f"MEDIUM risk: {advice}")
    else:
        st.success(f"LOW risk: {advice}")

    st.subheader("What-if analysis")
    st.write("See how the failure probability changes when one input varies and everything else stays fixed.")
    var = st.selectbox("Vary this input",
                       ["Operational_Hours", "Vibration_mms", "Temperature_C"])
    lo, hi = RANGES[var]
    grid = np.linspace(lo, hi, 60)
    sweep = pd.concat([row] * len(grid), ignore_index=True)
    sweep[var] = grid
    curve = pd.DataFrame({var: grid, "Failure probability": model.predict_proba(sweep)[:, 1]})
    st.line_chart(curve.set_index(var))

# ---------- Model info tab ----------
with tab_info:
    st.subheader("Model")
    st.write("Gradient Boosting classifier (scikit-learn) with one-hot encoding for machine type, "
             f"trained on 400,000 machines. Alert threshold: **{THRESHOLD:.2f}**.")

    st.subheader("Test-set performance (100,000 unseen machines)")
    st.table(pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"],
        "Value": [0.640, 0.807, 0.714, 0.983, 0.748],
    }).set_index("Metric"))

    st.subheader("What drives the prediction")
    st.write("Operational hours is by far the strongest factor. Vibration and temperature are the next most "
             "informative sensor signals, and they matter most for machines that are already old.")

    st.subheader("Limitations")
    st.write("The dataset is synthetic, so results show the approach, not real factory performance. "
             "Only about 6% of machines fail, so precision and recall are more meaningful than accuracy.")