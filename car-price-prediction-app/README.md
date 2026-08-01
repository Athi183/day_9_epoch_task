# Car Price Prediction App

This project is a Streamlit web application that predicts used car selling prices using a pre-trained scikit-learn pipeline (RandomForestRegressor + preprocessing).

## Files
- `app.py` – Streamlit application for entering car details and getting a prediction.
- `train_model.py` – Script to train the regression model and save it as `model.pkl`.
- `model.pkl` – Serialized scikit-learn pipeline (preprocessor + regressor).
- `cardekho_dataset.csv` – Dataset used for training / examples.
- `requirements.txt` – Python dependencies.

## Quickstart (local)
1. Create and activate a virtual environment (recommended).

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. (Optional) Retrain the model:

```bash
python train_model.py
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deployment

You can deploy this Streamlit app to platforms like Streamlit Community Cloud, Render, or Hugging Face Spaces. Typical steps:

- Push this repository to GitHub.
- Connect the repository to the hosting platform and set the start command to `streamlit run app.py`.

## Assignment Submission Template

When submitting for Epochs '26 — Assignment 9, include the following in your GitHub repo and submission message:

- **Participant Name:** _Your Name_
- **MUID:** _Your MUID_
- **Project overview:** Brief description and model used.
- **Deployment link:** Public URL to the hosted app.
- **Observations & Challenges:** Short reflection on deployment and model behavior.
- **Future improvements:** Ideas for improving the app or model.

## Notes
- The application expects the model file `model.pkl` to be present in the project root. If you retrain, `train_model.py` will overwrite `model.pkl`.
- Ensure realistic inputs for better predictions; the model was trained on a specific CarDekho dataset and may not generalize to all cars.
