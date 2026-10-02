"""FastAPI backend: loads the trained model, stores machines + prediction history in SQLite.
Run from the project root:  uvicorn backend.main:app --reload
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "models")
DB_PATH = ROOT / "machines.db"

model = joblib.load(MODELS_DIR / "model.pkl")
with open(MODELS_DIR / "config.json") as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

# Optional: a second model that predicts remaining useful life (RUL) in days.
# If models/rul_model.pkl does not exist, RUL is simply returned as null.
RUL_PATH = MODELS_DIR / "rul_model.pkl"
rul_model = joblib.load(RUL_PATH) if RUL_PATH.exists() else None


class Features(BaseModel):
    Machine_Type: str
    Operational_Hours: int
    Temperature_C: float
    Vibration_mms: float
    Sound_dB: float
    Oil_Level_pct: float
    Coolant_Level_pct: float
    Power_Consumption_kW: float
    Last_Maintenance_Days_Ago: int
    Maintenance_History_Count: int
    Failure_History_Count: int
    AI_Supervision: int
    Error_Codes_Last_30_Days: int
    Machine_Age: int


class PredictRequest(BaseModel):
    machine_code: str
    features: Features


class AuthIn(BaseModel):
    username: str
    password: str


class AccessIn(BaseModel):
    username: str
    machine_code: str


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _hash(pw, salt=None):
    salt = salt or os.urandom(8).hex()
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${h}"


def init_db():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)")
        con.execute("""CREATE TABLE IF NOT EXISTS access_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, machine_code TEXT NOT NULL,
            checked_at TEXT NOT NULL)""")
        con.execute("CREATE TABLE IF NOT EXISTS machines (machine_code TEXT PRIMARY KEY, data TEXT NOT NULL)")
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, machine_code TEXT NOT NULL,
            predicted_rul_days REAL, failure_probability REAL, failure_within_7_days INTEGER,
            risk_level TEXT, created_at TEXT NOT NULL)""")
        if con.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0:
            demo = dict(Machine_Type=config["machine_types"][0], Operational_Hours=50000,
                        Temperature_C=50.0, Vibration_mms=8.0, Sound_dB=70.0, Oil_Level_pct=70.0,
                        Coolant_Level_pct=70.0, Power_Consumption_kW=50.0,
                        Last_Maintenance_Days_Ago=180, Maintenance_History_Count=5,
                        Failure_History_Count=1, AI_Supervision=1,
                        Error_Codes_Last_30_Days=2, Machine_Age=5)
            con.execute("INSERT INTO machines VALUES (?, ?)", ("MC_TEST_001", json.dumps(demo)))


init_db()
app = FastAPI(title="Predictive Maintenance API")


def run_model(features: dict) -> dict:
    row = pd.DataFrame([features])[config["features"]]
    prob = float(model.predict_proba(row)[0, 1])
    rul = float(rul_model.predict(row)[0]) if rul_model is not None else None
    level = "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= THRESHOLD else "LOW"
    return {"failure_probability": round(prob, 4),
            "failure_within_7_days": prob >= THRESHOLD,
            "risk_level": level,
            "predicted_rul_days": None if rul is None else round(rul, 3)}


@app.get("/config")
def get_config():
    return config


@app.get("/machines")
def list_machines():
    with db() as con:
        return [r["machine_code"] for r in con.execute("SELECT machine_code FROM machines ORDER BY machine_code")]


@app.get("/machines/{code}")
def get_machine(code: str):
    with db() as con:
        r = con.execute("SELECT data FROM machines WHERE machine_code = ?", (code,)).fetchone()
    if r is None:
        raise HTTPException(404, "Machine not found")
    return json.loads(r["data"])


@app.post("/predict")
def predict(req: PredictRequest):
    feats = req.features.model_dump()
    if feats["Machine_Type"] not in config["machine_types"]:
        raise HTTPException(422, "Unknown machine type")
    result = run_model(feats)
    with db() as con:
        con.execute("INSERT OR REPLACE INTO machines VALUES (?, ?)", (req.machine_code, json.dumps(feats)))
        con.execute("""INSERT INTO predictions (machine_code, predicted_rul_days, failure_probability,
                       failure_within_7_days, risk_level, created_at) VALUES (?,?,?,?,?,?)""",
                    (req.machine_code, result["predicted_rul_days"], result["failure_probability"],
                     int(result["failure_within_7_days"]), result["risk_level"],
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return result


@app.get("/fleet")
def fleet():
    with db() as con:
        rows = con.execute("""SELECT p.machine_code, p.failure_probability, p.risk_level,
                p.predicted_rul_days, p.created_at, m.data
            FROM predictions p JOIN machines m ON m.machine_code = p.machine_code
            WHERE p.prediction_id IN (SELECT MAX(prediction_id) FROM predictions GROUP BY machine_code)"""
        ).fetchall()
    return [{"machine_code": r["machine_code"],
             "machine_type": json.loads(r["data"])["Machine_Type"],
             "failure_probability": r["failure_probability"], "risk_level": r["risk_level"],
             "predicted_rul_days": r["predicted_rul_days"], "created_at": r["created_at"]}
            for r in rows]


@app.get("/history/{code}")
def history(code: str):
    with db() as con:
        rows = con.execute("SELECT * FROM predictions WHERE machine_code = ? ORDER BY prediction_id DESC",
                           (code,)).fetchall()
    return [dict(r) for r in rows]


@app.post("/register")
def register(a: AuthIn):
    username = a.username.strip()
    if not username or not a.password:
        raise HTTPException(422, "Username and password required")
    with db() as con:
        if con.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise HTTPException(400, "Username already exists")
        con.execute("INSERT INTO users VALUES (?, ?)", (username, _hash(a.password)))
    return {"ok": True}


@app.post("/login")
def login(a: AuthIn):
    username = a.username.strip()
    with db() as con:
        row = con.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if not row or _hash(a.password, row["password_hash"].split("$")[0]) != row["password_hash"]:
        raise HTTPException(401, "Wrong username or password")
    return {"ok": True}


@app.post("/access_log")
def add_access(a: AccessIn):
    with db() as con:
        con.execute("INSERT INTO access_log (username, machine_code, checked_at) VALUES (?,?,?)",
                    (a.username, a.machine_code, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return {"ok": True}


@app.get("/access_log/{code}")
def get_access(code: str):
    with db() as con:
        rows = con.execute("SELECT username, machine_code, checked_at FROM access_log "
                           "WHERE machine_code = ? ORDER BY log_id DESC", (code,)).fetchall()
    return [dict(r) for r in rows]