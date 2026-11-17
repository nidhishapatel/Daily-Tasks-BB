import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Pharmacy Demand Predictor", layout="wide")
st.title("Pharmacy Sales Prediction & Demand Planning")

@st.cache_resource
def load_model_suite():
    model = joblib.load("sales_model.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("features.pkl")
    df = pd.read_csv("global_pharmacy_sales_data.csv")
    return model, encoders, features, df

model, encoders, features, df = load_model_suite()

tabs = st.tabs(["Real-Time Prediction", "Batch Prediction & Analysis", "Model Performance & Insights"])

with tabs[0]:
    st.subheader("Single Medicine Demand Forecaster")
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", encoders['category'].classes_)
        medicine = st.selectbox("Medicine", encoders['medicine'].classes_)
        age_group = st.selectbox("Age Group", encoders['age_group'].classes_)
    with col2:
        unit_price = st.number_input("Unit Price ($)", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
        stock_level = st.number_input("Current Stock Level", min_value=1, max_value=20000, value=4000, step=50)
    with col3:
        expiry_days = st.slider("Expiry Days Remaining", 30, 730, 365)
        covid_flag = st.selectbox("COVID Outbreak Surge Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Forecast Demand", key="rt_btn"):
        row = pd.DataFrame([{
            'category': encoders['category'].transform([category])[0],
            'medicine': encoders['medicine'].transform([medicine])[0],
            'age_group': encoders['age_group'].transform([age_group])[0],
            'unit_price': unit_price,
            'stock_level': stock_level,
            'expiry_days_remaining': expiry_days,
            'covid_flag': covid_flag
        }])
        pred_units = model.predict(row)[0]
        est_revenue = pred_units * unit_price

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Predicted Units Sold", f"{pred_units:.1f}")
        mcol2.metric("Projected Revenue", f"${est_revenue:,.2f}")
        stock_coverage = "Adequate Stock" if stock_level >= pred_units else "Stock-out Warning!"
        mcol3.metric("Inventory Status", stock_coverage)
        if stock_level < pred_units:
            st.warning(f"Reorder Recommended: Demand ({pred_units:.1f}) exceeds available inventory ({stock_level}).")

with tabs[1]:
    st.subheader("Batch Prediction Workflow")
    sample_csv = df[features].head(10).to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample Batch Template", sample_csv, "sample_batch_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Records Preview:", batch_df.head())
        if st.button("Run Batch Forecast"):
            work_df = batch_df.copy()
            for col in ['category', 'medicine', 'age_group']:
                if col in work_df.columns:
                    work_df[col] = encoders[col].transform(work_df[col].astype(str))
            batch_df['predicted_units_sold'] = model.predict(work_df[features]).round(2)
            if 'unit_price' in batch_df.columns:
                batch_df['projected_revenue'] = (batch_df['predicted_units_sold'] * batch_df['unit_price']).round(2)

            b1, b2, b3 = st.columns(3)
            b1.metric("Total Batch Records", len(batch_df))
            b2.metric("Total Predicted Units", f"{batch_df['predicted_units_sold'].sum():,.1f}")
            if 'projected_revenue' in batch_df.columns:
                b3.metric("Total Projected Revenue", f"${batch_df['projected_revenue'].sum():,.2f}")

            st.write("Prediction Results:", batch_df.head(10))
            st.bar_chart(batch_df['predicted_units_sold'].head(20))

            csv_result = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecasted CSV", csv_result, "batch_forecast_results.csv", "text/csv")

with tabs[2]:
    st.subheader("Model Performance & Feature Importance")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.write("### Feature Importance Ranking")
    st.bar_chart(importances)