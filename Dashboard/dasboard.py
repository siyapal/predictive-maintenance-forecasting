"""Streamlit frontend. Start backend first:  uvicorn backend.main:app --reload
Then:  streamlit run Dashboard/dasboard.py"""
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

st.markdown("""<style>
.block-container{padding-top:1.4rem}
.hero{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);padding:22px 28px;border-radius:16px;
      color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:1.7rem;color:#fff}.hero p{margin:4px 0 0;opacity:.8}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.22);
      padding:14px 16px;border-radius:14px}
.badge{display:inline-block;padding:8px 22px;border-radius:999px;font-weight:700;color:#fff;font-size:1.15rem}
.LOW{background:#16a34a}.MEDIUM{background:#f59e0b}.HIGH{background:#dc2626}
</style>""", unsafe_allow_html=True)

# ---------- Login ----------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown('<div class="hero"><h1>🛠️ Predictive Maintenance Login</h1>'
                '<p>Machine details dekhne ke liye pehle login karo.</p></div>', unsafe_allow_html=True)
    tab_login, tab_signup = st.tabs(["Login", "Sign up"])
    try:
        with tab_login:
            lu = st.text_input("Username", key="lu")
            lp = st.text_input("Password", type="password", key="lp")
            if st.button("Login", type="primary"):
                r = requests.post(f"{API}/login", json={"username": lu, "password": lp}, timeout=30)
                if r.ok:
                    st.session_state.user = lu.strip()
                    st.rerun()
                else:
                    st.error("Wrong username or password")
        with tab_signup:
            su = st.text_input("Choose username", key="su")
            sp = st.text_input("Choose password", type="password", key="sp")
            if st.button("Create account"):
                r = requests.post(f"{API}/register", json={"username": su, "password": sp}, timeout=30)
                if r.ok:
                    st.success("Account ban gaya, ab Login tab se login karo.")
                else:
                    st.error("Username already exists ya details khali hain.")
    except requests.exceptions.ConnectionError:
        st.error("Backend chal nahi raha. Pehle terminal mein chalao: uvicorn backend.main:app --reload")
    st.stop()


st.markdown('<div class="hero"><h1>🛠️ Predictive Maintenance and Failure Forecasting</h1>'
            '<p>Fleet health, 7-day failure risk and remaining useful life (RUL) in one place.</p></div>',
            unsafe_allow_html=True)


def api(method, path, **kw):
    r = requests.request(method, API + path, timeout=30, **kw)
    r.raise_for_status()
    return r.json()


try:
    cfg = api("GET", "/config")
except Exception:
    st.error("Backend chal nahi raha. Pehle terminal mein chalao: uvicorn backend.main:app --reload")
    st.stop()

RANGES = cfg["ranges"]
SENSORS = [("Temperature_C", "Temperature", "°C"), ("Sound_dB", "Sound", "dB"),
           ("Coolant_Level_pct", "Coolant Level", "%"), ("Vibration_mms", "Vibration", "mm/s"),
           ("Oil_Level_pct", "Oil Level", "%"), ("Power_Consumption_kW", "Power Consumption", "kW")]
INT_COLS = [("Operational_Hours", "Operational hours"), ("Machine_Age", "Machine age (years)"),
            ("Last_Maintenance_Days_Ago", "Days since last maintenance"),
            ("Maintenance_History_Count", "Past maintenance count"),
            ("Failure_History_Count", "Past failure count"),
            ("Error_Codes_Last_30_Days", "Error codes (30 days)")]
DEFAULTS = {"Operational_Hours": 50000, "Temperature_C": 60.0, "Vibration_mms": 10.0, "Sound_dB": 75.0,
            "Oil_Level_pct": 70.0, "Coolant_Level_pct": 65.0, "Power_Consumption_kW": 150.0,
            "Last_Maintenance_Days_Ago": 180, "Maintenance_History_Count": 5,
            "Failure_History_Count": 2, "Error_Codes_Last_30_Days": 3, "Machine_Age": 13}
ADVICE = {"HIGH": ("error", "HIGH risk: schedule maintenance now."),
          "MEDIUM": ("warning", "MEDIUM risk: inspect this machine soon."),
          "LOW": ("success", "LOW risk: normal operation.")}


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def sliders(base, key):
    out = dict(base)
    with st.sidebar.expander("Sensor and usage readings", expanded=(mode == "New Machine")):
        for col, label, unit in SENSORS:
            lo, hi = RANGES[col]
            out[col] = st.slider(f"{label} ({unit})", float(lo), float(hi),
                                 float(clamp(base.get(col, DEFAULTS[col]), lo, hi)), key=f"{key}_{col}")
        for col, label in INT_COLS:
            lo, hi = RANGES[col]
            out[col] = st.slider(label, int(lo), int(hi),
                                 int(clamp(base.get(col, DEFAULTS[col]), lo, hi)), key=f"{key}_{col}")
        ai = st.selectbox("AI supervision", ["No", "Yes"], index=int(base.get("AI_Supervision", 0)),
                          key=f"{key}_ai")
        out["AI_Supervision"] = 1 if ai == "Yes" else 0
    return out


# ---------- Sidebar ----------
st.sidebar.write(f"Logged in: **{st.session_state.user}**")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.pop("last_logged", None)
    st.rerun()
st.sidebar.divider()
st.sidebar.header("Machine Information")
mode = st.sidebar.radio("Input Mode", ["Existing Machine", "New Machine"])
if mode == "Existing Machine":
    codes = api("GET", "/machines")
    code = st.sidebar.selectbox("Select Machine (type karke search karo)", codes)
    m = sliders(api("GET", f"/machines/{code}"), code)
else:
    code = st.sidebar.text_input("Machine code", "MC_NEW_001").strip()
    mtype = st.sidebar.selectbox("Machine type", cfg["machine_types"])
    m = sliders({"Machine_Type": mtype}, "new")
predict_clicked = st.sidebar.button("Predict Machine Health", type="primary", width="stretch")


def log_access(machine_code):
    try:
        api("POST", "/access_log", json={"username": st.session_state.user, "machine_code": machine_code})
    except Exception:
        pass


# Existing machine: log once whenever a different machine is selected.
if mode == "Existing Machine" and code and st.session_state.get("last_logged") != code:
    log_access(code)
    st.session_state.last_logged = code

tab_fleet, tab_machine = st.tabs(["📊 Fleet Overview", "🔧 Machine Detail"])

# ---------- Fleet overview ----------
with tab_fleet:
    fleet = pd.DataFrame(api("GET", "/fleet"))
    if fleet.empty:
        st.info("Abhi koi prediction record nahi hai. Terminal mein `python seed_db.py` chalao "
                "ya Machine Detail tab mein predict karo.")
    else:
        # --- Selected machine card (badalta hai jab sidebar mein machine badlo) ---
        sel = fleet[fleet.machine_code == code]
        st.subheader(f"Selected machine: {code}")
        if sel.empty:
            st.info("Is machine ki abhi koi prediction nahi hai. Sidebar mein 'Predict Machine Health' dabao.")
            sel_type = m["Machine_Type"]
        else:
            s = sel.iloc[0]
            sel_type = s["machine_type"]
            rank = int((fleet.failure_probability > s["failure_probability"]).sum()) + 1
            k = st.columns(4)
            k[0].metric("Risk level", s["risk_level"])
            k[1].metric("Failure probability (7d)", f"{s['failure_probability']:.1%}")
            k[2].metric("Predicted RUL", "N/A" if pd.isna(s["predicted_rul_days"])
                        else f"{s['predicted_rul_days']:.0f} days")
            k[3].metric("Risk rank", f"#{rank} of {len(fleet)}", help="#1 = sabse zyada failure risk")

        # --- Scope: same type ya poori fleet ---
        scope = st.radio("Fleet summary dikhao", [f"Same type: {sel_type}", "Whole fleet"],
                         horizontal=True)
        view = fleet[fleet.machine_type == sel_type] if scope.startswith("Same type") else fleet

        k = st.columns(4)
        k[0].metric("Machines tracked", f"{len(view):,}")
        k[1].metric("🔴 High risk", int((view.risk_level == "HIGH").sum()))
        k[2].metric("🟡 Medium risk", int((view.risk_level == "MEDIUM").sum()))
        k[3].metric("Median RUL", "N/A" if view.predicted_rul_days.isna().all()
                    else f"{view.predicted_rul_days.median():.0f} days")
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Risk distribution")
            st.bar_chart(view.risk_level.value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0))
        with right:
            st.subheader("Machines needing attention first")
            top = view.sort_values("failure_probability", ascending=False).head(15).copy()
            top["Failure %"] = top.failure_probability * 100
            top["Selected"] = top.machine_code.eq(code).map({True: "◀ selected", False: ""})
            st.dataframe(
                top[["machine_code", "machine_type", "risk_level", "Failure %", "predicted_rul_days", "Selected"]],
                hide_index=True, width="stretch",
                column_config={"Failure %": st.column_config.ProgressColumn(
                    "Failure % (7d)", min_value=0, max_value=100, format="%.1f%%"),
                    "machine_code": "Machine", "machine_type": "Type", "risk_level": "Risk",
                    "predicted_rul_days": st.column_config.NumberColumn("RUL (days)", format="%.0f")})
        with st.expander("Poora fleet table dekho / download karo"):
            st.dataframe(view, hide_index=True, width="stretch")
            st.download_button("Download CSV", view.to_csv(index=False), "fleet_predictions.csv", "text/csv")

# ---------- Machine detail ----------
with tab_machine:
    st.subheader("Machine Overview")
    a = st.columns(4)
    a[0].metric("Machine Code", code)
    a[1].metric("Machine Type", m["Machine_Type"])
    a[2].metric("Installation Year", datetime.now().year - m["Machine_Age"])
    a[3].metric("Operational Hours", f"{m['Operational_Hours']:,}")
    b = st.columns(4)
    b[0].metric("Maintenance History", m["Maintenance_History_Count"])
    b[1].metric("Failure History", m["Failure_History_Count"])
    b[2].metric("Error Codes (30 Days)", m["Error_Codes_Last_30_Days"])
    b[3].metric("AI Supervision", "Enabled" if m["AI_Supervision"] else "Disabled")

    st.subheader("Current Machine Condition")
    cols = st.columns(3)
    for i, (col, label, unit) in enumerate(SENSORS):
        cols[i % 3].metric(label, f"{m[col]:.2f} {unit}")

    if predict_clicked:
        if not code:
            st.error("Machine code daalo.")
        else:
            res = api("POST", "/predict", json={"machine_code": code, "features": m})
            if mode == "New Machine":
                log_access(code)
            lvl, rul = res["risk_level"], res["predicted_rul_days"]
            st.divider()
            st.subheader("Prediction Result")
            r = st.columns(3)
            r[0].metric("Failure Probability (7 days)", f"{res['failure_probability']:.1%}")
            r[1].metric("Predicted RUL", "N/A" if rul is None else f"{rul:.0f} days")
            r[2].markdown(f'<p style="margin:0;opacity:.7;font-size:.9rem">Risk level</p>'
                          f'<span class="badge {lvl}">{lvl}</span>', unsafe_allow_html=True)
            st.progress(min(res["failure_probability"], 1.0))
            kind, text = ADVICE[lvl]
            getattr(st, kind)(text)

    st.divider()
    st.subheader("Prediction History")
    hist = pd.DataFrame(api("GET", f"/history/{code}")) if code else pd.DataFrame()
    if hist.empty:
        st.info("Is machine ki abhi koi prediction nahi hai. Sidebar mein 'Predict Machine Health' dabao.")
    else:
        latest = hist.iloc[0]
        h = st.columns(3)
        h[0].metric("Total Predictions", len(hist))
        h[1].metric("Latest RUL", "N/A" if pd.isna(latest["predicted_rul_days"])
                    else f"{latest['predicted_rul_days']:.0f} days")
        h[2].metric("Latest Failure Probability", f"{latest['failure_probability']:.1%}")
        trend = hist.sort_values("created_at").set_index("created_at")
        c1, c2 = st.columns(2)
        c1.markdown("**Failure probability trend**")
        c1.line_chart(trend["failure_probability"])
        if trend["predicted_rul_days"].notna().any():
            c2.markdown("**RUL trend (days)**")
            c2.line_chart(trend["predicted_rul_days"])
        if len(hist) < 2:
            st.caption("Trend line ke liye kam se kam 2 predictions chahiye. Sidebar mein koi reading badlo "
                       "aur dobara Predict dabao.")
        st.markdown("**Prediction records**")
        show = hist.drop(columns=["prediction_id"]).copy()
        show["failure_within_7_days"] = show["failure_within_7_days"].astype(bool)
        st.dataframe(show, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Access History")
    st.caption("Kisne, kab, is machine ko check kiya.")
    try:
        access = pd.DataFrame(api("GET", f"/access_log/{code}")) if code else pd.DataFrame()
    except Exception:
        access = pd.DataFrame()
    if access.empty:
        st.info("Is machine ko abhi kisi ne check nahi kiya.")
    else:
        access = access.rename(columns={"username": "User", "machine_code": "Machine", "checked_at": "Checked at"})
        st.dataframe(access, hide_index=True, width="stretch")
