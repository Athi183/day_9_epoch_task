# Epochs '26 Day 9 — Car Price Prediction (Streamlit)

This project is a compact, user-friendly Streamlit app that estimates the selling price of used cars. It was built as a practical exercise in end-to-end machine learning engineering: data preparation, model training, packaging a prediction pipeline, and deploying the result to a hosted service.

What you get in this repository

- A Streamlit-based web UI that collects common car attributes and returns a price estimate.
- A repeatable scikit-learn pipeline (preprocessing + Random Forest regressor).
- A training script and the original CarDekho CSV so the model can be regenerated.

Why this matters

This repo demonstrates how to move from a dataset to a shareable app. It focuses on reproducibility (training script), portability (a single serialized pipeline), and a simple UX so anyone can try the model without code.

Key files and layout

- `app.py` — root entrypoint (used by hosting platforms). It calls the Streamlit app in the subfolder.
- `car-price-prediction-app/app.py` — the main Streamlit UI and prediction code.
- `car-price-prediction-app/train_model.py` — training logic (reads `cardekho_dataset.csv`, fits the pipeline, writes `model.pkl`). The function `train_model()` returns the fitted pipeline when used programmatically.
- `car-price-prediction-app/cardekho_dataset.csv` — dataset used for training and fallback retraining.
- `car-price-prediction-app/model.pkl` — serialized model created by the training script (may be created at runtime if missing).
- `requirements.txt` & `car-price-prediction-app/requirements.txt` — pinned package lists for development and deployment.
- `render.yaml` and `runtime.txt` — simple Render configuration used for deployment.

How the app behaves (practical summary)

1. On startup the app looks for `car-price-prediction-app/model.pkl`.
2. If the pickle is missing or cannot be loaded (scikit-learn version mismatch), the app automatically retrains a fresh model from the CSV and uses that model for predictions.
3. The UI exposes inputs for vehicle age (derived from year), kilometers driven, fuel type, seller type, transmission, mileage, engine size, max power and seats.

Local development — quick commands

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run only the sub-app (optional):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r car-price-prediction-app/requirements.txt
streamlit run car-price-prediction-app/app.py
```

Deployment

The project is deployed on Render at:

`https://day-9-epoch-task.onrender.com/`

If you deploy elsewhere, ensure the Python runtime and pinned package versions match the environment used when the model was created. Version mismatches (especially scikit-learn) are the most common cause of pickle-load failures.

Design and implementation notes (short)

- Preprocessing pipeline: `ColumnTransformer` is used to apply a median imputer to numeric columns and a simple imputer + `OneHotEncoder` to categorical columns. This is followed by a `RandomForestRegressor`.
- Reproducibility: `train_model.py` uses a fixed `random_state` and a train/test split so you can evaluate the model and reproduce the reported MAE.
- Robust serving: the app detects pickle incompatibility and retrains from the CSV in the running environment to recover safely.

Challenges we faced and how they were resolved

- Build-time wheels / runtime mismatch: the hosting platform initially used a Python version whose binary wheels (Pillow, etc.) did not match, causing build failures. We pinned Python (via `runtime.txt`) and updated key package versions to match available wheels.
- scikit-learn pickles: model files saved under one scikit-learn release can fail to load with another. To address this we:
  - pinned `scikit-learn` in `requirements.txt` for reproducibility, and
  - added a recovery path that retrains automatically when loading fails.

Notes on `model.pkl` and version control

Serialized models can be large. For quick experiments it's convenient to keep `model.pkl` in the repo, but for production workflows consider storing models in a dedicated model registry, cloud storage, or release assets instead of tracking them in Git.

Author

- **Name:** Athira V
- **Mulearn ID:** athirav-3@mulearn

Next steps I can help with

- Add an author card to the app sidebar.
- Add a short "Data preparation" section listing columns used and any simple cleaning steps.
- Create a tiny CONTRIBUTING note explaining how to retrain and push a new model to a release asset.

If you'd like any of those, tell me which and I’ll implement them.
