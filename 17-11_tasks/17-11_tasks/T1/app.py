import streamlit as st
import pandas as pd
import joblib

st.title("Pharmacy Sales Analytics and Prediction")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head(10))
    
    required = ['year', 'month', 'stock_level', 'expiry_days_remaining', 'covid_flag', 'unit_price']
    missing = [c for c in required if c not in data.columns]
    
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        st.success("Dataset validated successfully")
        
        if st.button("Generate Demand Predictions"):
            model = joblib.load('demand_model.pkl')
            data['predicted_units_sold'] = model.predict(data[required])
            st.subheader("Predictions Output")
            st.dataframe(data[['medicine', 'category', 'stock_level', 'predicted_units_sold']].head(20))