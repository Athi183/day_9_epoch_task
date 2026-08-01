# Epochs '26 Day 9 — Used Car Price Prediction

This repository contains a deployment-ready Streamlit application built for the Epochs '26 Day 9 assignment. The app predicts used car selling prices using a trained scikit-learn regression model on the CarDekho dataset.

## Project Overview

The app is designed to accept car attributes from the user, preprocess them, and predict the expected selling price. It supports the following workflow:

- Load the trained model pipeline from `car-price-prediction-app/model.pkl` if available.
- Otherwise, train a new model from `car-price-prediction-app/cardekho_dataset.csv`.
- Collect user inputs for features like year, kilometers driven, fuel type, seller type, transmission, owner type, and more.
- Preprocess categorical and numerical features using scikit-learn transformers.
- Display the price prediction in the Streamlit UI.

## Author

- **Name:** Athira V
- **Mulearn ID:** athirav-3@mulearn


## Repository Structure

- `app.py` — root Streamlit entrypoint used by deployment platforms.
- `car-price-prediction-app/app.py` — primary application code and UI layout.
- `car-price-prediction-app/train_model.py` — training script to build the model and serialize it.
- `car-price-prediction-app/cardekho_dataset.csv` — dataset for training and fallback retraining.
- `car-price-prediction-app/requirements.txt` — app-specific dependency pins.
- `requirements.txt` — root dependency file for development and deployment.
- `render.yaml` — Render service configuration.
- `runtime.txt` — Python runtime pin for Render.

## Features

- Predicts used car selling price with a scikit-learn model pipeline.
- Supports user-friendly Streamlit controls for categorical and numeric inputs.
- Handles missing serialized model by retraining automatically.
- Includes deployment-ready configuration for Render.

## Deployment

### Render

The app is deployed on Render at:

`https://day-9-epoch-task.onrender.com/`

### Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If you prefer to run the app directly from the subfolder:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r car-price-prediction-app/requirements.txt
streamlit run car-price-prediction-app/app.py
```

## Challenges Encountered

- Python runtime compatibility: Render initially built the app on Python 3.14, which caused package compatibility issues. A `runtime.txt` pin and updated `render.yaml` were used to stabilize deployment.
- Dependency resolution: Streamlit and Pillow version conflicts prevented a clean install on the Render build image. Upgrading Streamlit to a compatible version resolved the wheel install issue.
- Model serialization: Pre-trained model loading required a consistent scikit-learn/joblib environment. The model pipeline was retrained and saved with pinned versions to ensure predictable behavior.
- Deployment configuration: The project needed both root and subfolder entrypoints plus a correct dependency file so Render and Streamlit would start the right app.

## Notes

- `model.pkl` is stored in `car-price-prediction-app/model.pkl` when available.
- If the saved model is missing or incompatible with the current scikit-learn environment, the app now automatically recreates the model from `car-price-prediction-app/cardekho_dataset.csv`.
- The app is built with scikit-learn, pandas, numpy, joblib, and Streamlit.
- The current deployment is confirmed working at the Render URL above.
