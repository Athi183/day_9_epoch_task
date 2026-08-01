from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).with_name("cardekho_dataset.csv")
MODEL_PATH = Path(__file__).with_name("model.pkl")


def train_model():
    df = pd.read_csv(DATA_PATH)
    # Use columns that exist in cardekho_dataset.csv
    features = [
        "vehicle_age",
        "km_driven",
        "fuel_type",
        "seller_type",
        "transmission_type",
        "mileage",
        "engine",
        "max_power",
        "seats",
    ]

    X = df[features]
    y = df["selling_price"]

    categorical_features = ["fuel_type", "seller_type", "transmission_type"]
    numerical_features = [col for col in features if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=120, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Training complete. Mean absolute error: {mae:,.0f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
