import streamlit as st
import pandas as pd
import joblib

# Load the trained base model
@st.cache_resource
def load_model():
    return joblib.load("base_model.pkl")

model = load_model()

# App title
st.title("Commercial Mispricing Predictor")

st.markdown("""
This tool predicts whether a sales transaction is likely to be **mispriced** — 
based on factors like discount, cost, quantity, product type, region, and customer segment.

A mispriced transaction is defined as:
- **Revenue > $100**
- **Profit margin < 5%**
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with sales transaction data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Make predictions
    predictions = model.predict(df)
    df["Prediction"] = predictions
    df["Prediction Label"] = df["Prediction"].map({0: "✅ Good Deal", 1: "🚨 Mispriced"})

    st.subheader("Prediction Summary")
    st.write(df["Prediction Label"].value_counts())

    st.subheader("Detailed Output")
    st.write(df)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, file_name="mispricing_predictions.csv", mime="text/csv")
