import streamlit as st
import pandas as pd
import json
import hashlib
import os
import joblib

st.set_page_config(page_title="Scalable Pharma Cloud Dashboard", layout="wide")

def verify_credentials(user, pwd):
    if not os.path.exists('users.json'):
        return False, None
    with open('users.json', 'r') as f:
        db = json.load(f)
    hashed = hashlib.sha256(pwd.encode()).hexdigest()
    if user in db and db[user]['password'] == hashed:
        return True, db[user]['role']
    return False, None

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user'] = ''
    st.session_state['role'] = ''

if not st.session_state['authenticated']:
    st.title("Secure Pharmacy Cloud Portal")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("User Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Sign In"):
            success, role = verify_credentials(u, p)
            if success:
                st.session_state['authenticated'] = True
                st.session_state['user'] = u
                st.session_state['role'] = role
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col2:
        st.info("Test Accounts: Admin: admin / Admin@Secure2026 | Manager: manager / Manager@Pharma123 | Analyst: analyst / Analyst@Data456")
else:
    user_role = st.session_state['role']
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Logged In:** `{st.session_state['user']}`")
    st.sidebar.markdown(f"**Role:** `{user_role.upper()}`")
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['user'] = ''
        st.session_state['role'] = ''
        st.rerun()

    # Role 1: Admin Dashboard
    if user_role == 'admin':
        st.title("Admin - Cloud Infrastructure & Cluster Health")
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cluster Status", "Healthy (3/3 Nodes)")
        c2.metric("Nginx Load Balancer", "Online (Port 80)")
        c3.metric("CPU Utilization", "32.4 %")
        c4.metric("Active Sessions", "3 Users")

        st.subheader("Cloud Node Replicas")
        nodes_df = pd.DataFrame([
            {"Node": "app_instance_1", "Port": 8501, "Status": "Running", "Latency": "14 ms", "Handled Requests": 48200},
            {"Node": "app_instance_2", "Port": 8501, "Status": "Running", "Latency": "18 ms", "Handled Requests": 47100},
            {"Node": "app_instance_3", "Port": 8501, "Status": "Running", "Latency": "16 ms", "Handled Requests": 47200}
        ])
        st.dataframe(nodes_df, use_container_width=True)

        st.subheader("System Security & User Accounts")
        with open('users.json', 'r') as f:
            user_data = json.load(f)
        users_table = [{"Username": k, "Role": v['role'], "Description": v['description']} for k, v in user_data.items()]
        st.table(users_table)

    # Role 2: Manager Dashboard
    elif user_role == 'manager':
        st.title("Manager - Operations & Revenue Performance")
        st.markdown("---")
        summary_data = pd.read_csv('monthly_summary.csv')
        total_rev = summary_data['revenue'].sum() if 'revenue' in summary_data.columns else 0
        total_units = summary_data['units_sold'].sum()
        total_stock = summary_data['stock_level'].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Revenue ($)", f"{total_rev:,.2f}")
        m2.metric("Total Units Sold", f"{total_units:,}")
        m3.metric("Total Current Stock", f"{total_stock:,}")

        st.subheader("Regional Sales Volume")
        reg_sales = summary_data.groupby('region')['units_sold'].sum()
        st.bar_chart(reg_sales)

        st.subheader("Monthly Regional Summary Table")
        st.dataframe(summary_data.head(25), use_container_width=True)

    # Role 3: Analyst Dashboard
    elif user_role == 'analyst':
        st.title("Analyst - Demand Forecasting & Inventory Risk")
        st.markdown("---")
        st.subheader("Interactive ML Demand Inference")
        col_a, col_b = st.columns(2)
        with col_a:
            stock = st.slider("Stock Level", 500, 10000, 3500)
            expiry = st.slider("Expiry Days Remaining", 10, 720, 250)
        with col_b:
            price = st.number_input("Unit Price ($)", 1.0, 500.0, 45.0)
            covid = st.selectbox("Covid Pandemic Flag", [0, 1])

        if st.button("Generate Cloud Model Prediction"):
            if os.path.exists('cloud_model.pkl'):
                model = joblib.load('cloud_model.pkl')
                pred_units = model.predict([[stock, expiry, price, covid]])[0]
                st.success(f"**Predicted Units Sold:** {round(pred_units, 1)} units")
            else:
                st.warning("Model file not found.")

        st.subheader("Historical Regional & Monthly Trends")
        summary_data = pd.read_csv('monthly_summary.csv')
        st.dataframe(summary_data, use_container_width=True)