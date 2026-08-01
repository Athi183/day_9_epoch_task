import datetime
import streamlit as st
import pandas as pd
import joblib


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")


# ----------------------------
# Load model
# ----------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    import sklearn

    skl_ver = getattr(sklearn, "__version__", "unknown")
    st.error(f"Could not load model: {e}")
    st.markdown(f"**Detected scikit-learn version:** {skl_ver}")
    st.markdown(
        "The saved model was likely created with a different scikit-learn version.\n\n"
        "To fix this, install the matching scikit-learn version in the Python environment used to run Streamlit and restart the app."
    )
    st.code("pip install scikit-learn==1.4.3", language="bash")
    st.markdown("If you're using a virtual environment activate it first (for Windows PowerShell):")
    st.code(".\\.venv\\Scripts\\Activate.ps1\npython -m pip install scikit-learn==1.4.3", language="powershell")
    st.stop()


# ----------------------------
# Title + About
# ----------------------------
st.title("🚗 Car Price Prediction")
st.write("Enter the car details below and click Predict to get an estimated selling price.")
st.divider()

st.sidebar.header("About")
st.sidebar.write(
    """
    This Streamlit app loads a pre-trained scikit-learn pipeline and predicts
    used car selling prices (CarDekho dataset). Built for Epochs '26 — Assignment 9.

    - Model: RandomForest pipeline with preprocessing
    - Enter realistic values for the best results
    """
)


# ----------------------------
# User inputs (match training features)
# Expected features: year, km_driven, fuel, seller_type, transmission, owner,
#                    mileage, engine, max_power, seats
# ----------------------------
current_year = datetime.date.today().year

year = st.number_input("Year of Registration", min_value=1950, max_value=current_year, value=current_year - 5)

km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)


fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"]) 

transmission = st.selectbox("Transmission", ["Manual", "Automatic"]) 

mileage = st.number_input("Mileage (kmpl / km/kg)", min_value=0.0, value=18.0, format="%.2f")

engine = st.number_input("Engine (CC)", min_value=100, max_value=6000, value=1197)

max_power = st.number_input("Max Power (bhp)", min_value=10.0, value=82.0, format="%.2f")

seats = st.number_input("Seats", min_value=1, max_value=10, value=5)


# ----------------------------
# Make prediction
# ----------------------------
if st.button("Predict Selling Price"):
    # model expects vehicle_age (not year)
    vehicle_age = current_year - int(year)

    input_df = pd.DataFrame(
        [
            {
                "vehicle_age": int(vehicle_age),
                "km_driven": int(km_driven),
                "fuel_type": fuel,
                "seller_type": seller_type,
                "transmission_type": transmission,
                "mileage": float(mileage),
                "engine": float(engine),
                "max_power": float(max_power),
                "seats": int(seats),
            }
        ]
    )

    try:
        preds = model.predict(input_df)
        price = preds[0]
        st.success(f"Estimated Selling Price: ₹ {price:,.2f}")
        st.info("Tip: Predictions are only as good as the input values and the training data.")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Make sure the input fields match the model's expected features and types.")