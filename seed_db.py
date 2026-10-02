"""Fills the database with real machines from your dataset (one first prediction each).
Backend chalu hona chahiye.  Run from project root:  python seed_db.py"""
import json
from pathlib import Path

import pandas as pd
import requests

API = "http://127.0.0.1:8000"
N = 300   # kitni machines load karni hain

ROOT = Path(__file__).resolve().parent
DATA = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "data")
ids = pd.read_csv(next(DATA.rglob("factory_sensor_data.csv.csv")), usecols=["Machine_ID"])["Machine_ID"]
clean = pd.read_csv(next(DATA.rglob("cleaned_data.csv")))   # same row order as the raw file
cfg = requests.get(API + "/config", timeout=15).json()

done = 0
for i in clean.sample(N, random_state=42).index:
    feats = json.loads(clean.loc[[i], cfg["features"]].to_json(orient="records"))[0]
    r = requests.post(API + "/predict", json={"machine_code": ids[i], "features": feats}, timeout=30)
    if r.ok:
        done += 1
    else:
        print("Failed", ids[i], r.status_code, r.text[:120])
print(f"Done: {done}/{N} machines saved with their first prediction.")