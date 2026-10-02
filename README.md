# Predictive Maintenance and Failure Forecasting

A machine-learning system that predicts, for each industrial machine:

- **Failure probability** within the next 7 days
- **Remaining Useful Life (RUL)** in days
- **Risk level**: LOW, MEDIUM or HIGH

It has a FastAPI backend, a SQLite database and a Streamlit dashboard with login and an access history log.

## Features

- Login and sign up (passwords stored as salted PBKDF2 hashes)
- Fleet Overview: risk distribution, machines needing attention first, summary by machine type or whole fleet
- Machine Detail: sensor readings, live prediction, prediction history with trend charts
- Access History: which user checked which machine, and when
- Add new machines and re-run predictions from the sidebar

## Project structure

```
.
├── backend/
│   └── main.py          # FastAPI app (model loading, SQLite, auth, endpoints)
├── Dashboard/
│   └── dasboard.py      # Streamlit frontend
├── Models/              # model.pkl, rul_model.pkl, config.json
├── Data/                # raw and processed data
├── train_rul.py         # trains the RUL model
├── seed_db.py           # fills the database with machines and first predictions
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS: `source venv/bin/activate`

## One-time preparation

```powershell
python train_rul.py     # creates Models/rul_model.pkl
python seed_db.py       # creates machines.db with machines and first predictions
```

## Run

Use two terminals, both with the virtual environment activated.

Terminal 1, backend:

```powershell
uvicorn backend.main:app --reload
```

Terminal 2, dashboard (start after the backend):

```powershell
streamlit run Dashboard\dasboard.py
```

Open http://localhost:8501, sign up, then log in.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/config` | Model config and feature ranges |
| GET | `/machines` | List machine codes |
| GET | `/machines/{code}` | Latest readings of a machine |
| POST | `/predict` | Run prediction and save to history |
| GET | `/fleet` | Latest prediction of every machine |
| GET | `/history/{code}` | Prediction history of a machine |
| POST | `/register`, `/login` | Authentication |
| POST | `/access_log` | Record that a user viewed a machine |
| GET | `/access_log/{code}` | Access history of a machine |

Interactive API docs: http://127.0.0.1:8000/docs

## Notes

- The login is basic and meant for a college or demo project. For real deployment add HTTPS and token-based authentication (for example JWT).
- `machines.db` is created locally and is not part of the repository.
- RUL is an estimate from sensor patterns. Read it together with the risk level.

## Tech stack

Python, FastAPI, SQLite, scikit-learn, pandas, Streamlit