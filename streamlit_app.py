import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Sales Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")

model = load_model()

st.title("Sales Prediction Model")
st.write("Adjust the inputs below and click **Predict** to get a sales forecast.")

col1, col2 = st.columns(2)

with col1:
    marketing_spend = st.slider("Marketing Spend ($)", 1000, 5000, 3000, step=100)
    discount = st.slider("Discount (%)", 0.0, 30.0, 10.0, step=0.5)
    foot_traffic = st.slider("Foot Traffic", 100, 1000, 500, step=10)

with col2:
    competitor_price = st.slider("Competitor Price ($)", 10.0, 50.0, 30.0, step=0.5)
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        index=2,
    )
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][x - 1],
        index=0,
    )

if st.button("Predict", type="primary"):
    features = np.array([[marketing_spend, discount, foot_traffic, competitor_price, day_of_week, month]])
    prediction = model.predict(features)[0]
    st.metric("Predicted Sales", f"${prediction:,.2f}")

# --- Batch prediction ---
st.divider()
st.subheader("Batch Prediction")
uploaded = st.file_uploader("Upload a CSV with columns: marketing_spend, discount, foot_traffic, competitor_price, day_of_week, month", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    required = ["marketing_spend", "discount", "foot_traffic", "competitor_price", "day_of_week", "month"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        df["predicted_sales"] = model.predict(df[required]).round(2)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")
