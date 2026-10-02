"""Trains the RUL (remaining useful life) model and saves Models/rul_model.pkl.
Run from the project root (jahan machines.db hai):  python train_rul.py
Phir backend restart karo (uvicorn ko Ctrl+C karke dobara chalao)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
MODELS = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "models")
DATA = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "data")

cfg = json.load(open(MODELS / "config.json"))
df = pd.read_csv(next(DATA.rglob("cleaned_data.csv")))
X, y = df[cfg["features"]], df["Remaining_Useful_Life_days"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Machine_Type"])],
                               remainder="passthrough")),
    ("reg", HistGradientBoostingRegressor(random_state=42)),
])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
print(f"Test R2: {r2_score(yte, pred):.3f} | MAE: {mean_absolute_error(yte, pred):.1f} days")
joblib.dump(pipe, MODELS / "rul_model.pkl")
print("Saved:", MODELS / "rul_model.pkl")