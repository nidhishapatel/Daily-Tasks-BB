import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib

st.set_page_config(page_title="Live Stream Pharmacy Analytics", layout="wide")
st.title("Real-Time Pharmacy Streaming Analytics & On-The-Fly Prediction")

@st.cache_resource
def load_stream_artifacts():
    model = joblib.load("stream_model.pkl")
    encoders = joblib.load("stream_encoders.pkl")
    features = joblib.load("stream_features.pkl")
    df = pd.read_csv("global_pharmacy_sales_data.csv")
    return model, encoders, features, df

model, encoders, features, df = load_stream_artifacts()

st.sidebar.header("Stream Configuration")
stream_speed = st.sidebar.slider("Streaming Speed (seconds)", 0.2, 2.0, 0.5)
stream_events = st.sidebar.slider("Number of Stream Events", 10, 60, 25)
selected_cat = st.sidebar.selectbox("Filter Stream by Category", ["All"] + list(df['category'].unique()))
start_stream = st.sidebar.button("Launch Live Stream Pipeline")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1_box = kpi1.empty()
kpi2_box = kpi2.empty()
kpi3_box = kpi3.empty()
kpi4_box = kpi4.empty()

chart_container = st.empty()
alert_container = st.empty()
table_container = st.empty()

if start_stream:
    stream_source = df if selected_cat == "All" else df[df['category'] == selected_cat]
    stream_records = []

    for i in range(stream_events):
        sample_row = stream_source.sample(1).iloc[0]
        cat_code = encoders['category'].transform([sample_row['category']])[0]
        med_code = encoders['medicine'].transform([sample_row['medicine']])[0]

        feat_df = pd.DataFrame([{
            'category': cat_code,
            'medicine': med_code,
            'unit_price': float(sample_row['unit_price']),
            'stock_level': int(sample_row['stock_level']),
            'covid_flag': int(sample_row['covid_flag'])
        }])

        pred_units = model.predict(feat_df)[0]
        actual_units = int(sample_row['units_sold'])
        stock = int(sample_row['stock_level'])
        is_low_stock = pred_units > stock

        stream_records.append({
            'Tick': i + 1,
            'Timestamp': sample_row['date'],
            'Category': sample_row['category'],
            'Medicine': sample_row['medicine'],
            'Actual Units': actual_units,
            'Predicted Units': round(float(pred_units), 2),
            'Stock Level': stock,
            'Stock Alert': "CRITICAL" if is_low_stock else "Normal"
        })

        curr_stream_df = pd.DataFrame(stream_records)

        kpi1_box.metric("Events Ingested", len(curr_stream_df))
        kpi2_box.metric("Latest Prediction", f"{pred_units:.1f} units")
        kpi3_box.metric("Avg Predicted Demand", f"{curr_stream_df['Predicted Units'].mean():.1f}")
        critical_count = (curr_stream_df['Stock Alert'] == "CRITICAL").sum()
        kpi4_box.metric("Stock Alerts", critical_count)

        chart_container.line_chart(curr_stream_df.set_index('Tick')[['Actual Units', 'Predicted Units']])

        if is_low_stock:
            alert_container.warning(f"Stock Alert on Event #{i+1}: {sample_row['medicine']} demand ({pred_units:.1f}) exceeds stock ({stock})!")

        table_container.dataframe(curr_stream_df.tail(6), use_container_width=True)
        time.sleep(stream_speed)