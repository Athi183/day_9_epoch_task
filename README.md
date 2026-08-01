# day_9_epoch_task

This repository contains the deployment-ready Streamlit app for the Epochs '26 Day 9 assignment. The app predicts used car selling prices using a scikit-learn model trained on the CarDekho dataset.

## Deployment

The root `app.py` file is the Streamlit entrypoint. When deployed, Streamlit will run this file and execute the app located in `car-price-prediction-app/app.py`.

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Click **New app** and choose the repository `Athi183/day_9_epoch_task`.
4. Set the main file path to `app.py`.
5. Deploy.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The trained model file `car-price-prediction-app/model.pkl` is intentionally ignored in Git because it exceeds GitHub's file size limit.
- If `model.pkl` is missing, the app will retrain the model automatically using `car-price-prediction-app/cardekho_dataset.csv`.
