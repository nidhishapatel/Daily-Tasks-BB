import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt

st.set_page_config(page_title="Pharmacy Visual Analytics Dashboard", layout="wide")
st.title("Pharmacy Global Sales & Insights Dashboard")

@st.cache_data
def load_data():
    data = pd.read_csv("global_pharmacy_sales_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data['revenue'] = data['units_sold'] * data['unit_price']
    data['year_month'] = data['date'].dt.to_period('M').astype(str)
    return data

df = load_data()

st.sidebar.header("Interactive Dashboard Filters")
min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

all_regions = list(df['region'].unique())
selected_regions = st.sidebar.multiselect("Select Regions", options=all_regions, default=all_regions[:4])

all_categories = ["All"] + list(df['category'].unique())
selected_category = st.sidebar.selectbox("Filter by Category", options=all_categories)

all_age_groups = ["All"] + list(df['age_group'].unique())
selected_age_group = st.sidebar.selectbox("Filter by Age Group", options=all_age_groups)

filtered_df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
if selected_regions:
    filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df['category'] == selected_category]
if selected_age_group != "All":
    filtered_df = filtered_df[filtered_df['age_group'] == selected_age_group]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Filtered Records", f"{len(filtered_df):,}")
k2.metric("Total Revenue", f"${filtered_df['revenue'].sum():,.2f}")
k3.metric("Total Units Sold", f"{filtered_df['units_sold'].sum():,}")
k4.metric("Avg Unit Price", f"${filtered_df['unit_price'].mean():.2f}")
k5.metric("Avg Stock Level", f"{int(filtered_df['stock_level'].mean()):,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Product & Regional Breakdown",
    "Pricing & Inventory Insights",
    "Data Explorer & Export"
])

with tab1:
    st.subheader("Monthly Revenue Trends Over Time (Plotly)")
    monthly_trend = filtered_df.groupby('year_month')['revenue'].sum().reset_index()
    fig_line = px.line(monthly_trend, x='year_month', y='revenue', title="Monthly Revenue Trajectory", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.subheader("Regional & Category Distribution (Plotly)")
    col_a, col_b = st.columns(2)
    with col_a:
        reg_summary = filtered_df.groupby('region')['units_sold'].sum().reset_index()
        fig_bar = px.bar(reg_summary, x='region', y='units_sold', color='region', title="Total Units Sold by Region")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_b:
        cat_summary = filtered_df.groupby('category')['revenue'].sum().reset_index()
        fig_pie = px.pie(cat_summary, values='revenue', names='category', title="Revenue Share by Category")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.subheader("Interactive Pricing vs Demand (Altair)")
    sample_size = min(1500, len(filtered_df))
    chart_sample = filtered_df.sample(sample_size, random_state=42)
    altair_scatter = alt.Chart(chart_sample).mark_circle(size=70).encode(
        x='unit_price:Q',
        y='units_sold:Q',
        color='category:N',
        tooltip=['medicine:N', 'unit_price:Q', 'units_sold:Q', 'region:N', 'age_group:N']
    ).interactive().properties(title="Unit Price vs Units Sold Dynamics", height=400)
    st.altair_chart(altair_scatter, use_container_width=True)

    st.subheader("Stock Level vs Demand by Category (Plotly)")
    fig_scatter = px.scatter(
        chart_sample,
        x='stock_level',
        y='units_sold',
        color='category',
        size='unit_price',
        hover_data=['medicine', 'region'],
        title="Stock Level vs Units Sold (Size = Price)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    st.subheader("Filtered Dataset Preview & Data Export")
    st.dataframe(filtered_df.head(25), use_container_width=True)
    csv_download = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data (CSV)", csv_download, "pharmacy_filtered_data.csv", "text/csv")