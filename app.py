import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up page config
st.set_page_config(page_title="Retail Demand Forecaster", layout="centered", page_icon="📈")
st.title("📈 Retail Sales & Demand Prediction UI")

# Load model pipeline securely
@st.cache_resource
def load_model_pipeline():
    return joblib.load('rf_demand_model.pkl')

artifacts = load_model_pipeline()
model = artifacts['model']
features = artifacts['features']
# =====================================================================
# 2. LOAD MODEL ARTIFACTS AND CLEAN DATA AUTOMATICALLY
# =====================================================================
@st.cache_resource
def load_model_pipeline():
    return joblib.load('rf_demand_model.pkl')

@st.cache_data
def load_local_dataset():
    # Automatically reads the CSV you just exported from Jupyter
    return pd.read_csv('cleaned_engineered_retail_data.csv')

try:
    artifacts = load_model_pipeline()
    model = artifacts['model']
    features = artifacts['features']
    
    # Load your historical data quietly on startup
    historical_df = load_local_dataset()
except FileNotFoundError as e:
    st.error(f"Missing File Error: {e.filename} not found in this folder. Please verify your Jupyter exports.")
    st.stop()

# Add a toggle view at the top of the UI so users can audit the data
st.write("### 📊 Live Master Inventory Database")
with st.expander("🔍 Click to inspect historical data records directly from Jupyter"):
    st.write(f"Total Database Rows: **{len(historical_df):,}** items logged.")
    st.dataframe(historical_df.head(10))

# =====================================================================
# 3. INTERACTIVE LAYOUT STRUCTURES (Updated with Item Details)
# =====================================================================
st.header("📦 Step 1: Product Identification & Attributes")

# New fields added to track the specific item being forecasted
col_id1, col_id2 = st.columns(2)
with col_id1:
    input_stock_code = st.text_input("Enter Item Stock Code", value="22423", help="e.g., 22423, 85123A")
with col_id2:
    input_description = st.text_input("Enter Product Description", value="REGENCY CAKESTAND 3 TIER")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    unit_price = st.number_input("Unit Price ($)", min_value=0.01, value=2.50, step=0.10)
    month = st.slider("Month of the Year", min_value=1, max_value=12, value=7)
    day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 6], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"][x if x < 5 else 5])

with col2:
    is_uk = st.selectbox("Customer Location", options=[1, 0], format_func=lambda x: "United Kingdom (Domestic)" if x == 1 else "International")
    day_of_year = st.slider("Day of the Year (1-365)", min_value=1, max_value=365, value=200)
    quarter = st.radio("Financial Quarter", options=[1, 2, 3, 4], horizontal=True, index=2)

st.header("👤 Step 2: Customer Lifetime Profile (RFM)")
col3, col4 = st.columns(2)

with col3:
    recency = st.number_input("Customer Recency (Days since last order)", min_value=0, value=15)
    frequency = st.number_input("Customer Frequency (Total past unique orders)", min_value=1, value=5)

with col4:
    monetary = st.number_input("Customer Monetary ($ Total past spend)", min_value=0.0, value=250.0)
    tenure = st.number_input("Customer Tenure (Days active in system)", min_value=1, value=120)

# Preprocessing features math transformations (remains same as your pipeline)
unit_price_log = np.log1p(unit_price)
is_weekend = 1 if day_of_week in [5, 6] else 0
month_sin = np.sin(2 * np.pi * month / 12.0)
month_cos = np.cos(2 * np.pi * month / 12.0)
cust_recency_sqrt = np.sqrt(recency)
cust_frequency_log = np.log1p(frequency)
cust_monetary_log = np.log1p(monetary)
cust_lifetime_value_log = np.log1p(monetary) 
cust_lifetime_orders_log = np.log1p(frequency)
cust_tenure_days_sqrt = np.sqrt(tenure)

input_data = pd.DataFrame([{
    'UnitPrice_Log': unit_price_log, 'Month': month, 'Day_of_Week': day_of_week, 'Is_Weekend': is_weekend, 'Is_UK': is_uk,
    'Cust_Recency_Sqrt': cust_recency_sqrt, 'Cust_Frequency_Log': cust_frequency_log, 'Cust_Monetary_Log': cust_monetary_log,
    'Day_of_Year': day_of_year, 'Quarter': quarter, 'Month_Sin': month_sin, 'Month_Cos': month_cos,
    'Cust_Lifetime_Value_Log': cust_lifetime_value_log, 'Cust_Lifetime_Orders_Log': cust_lifetime_orders_log, 'Cust_Tenure_Days_Sqrt': cust_tenure_days_sqrt
}])
input_data = input_data[features]
# =====================================================================
# 5. RUN FORECAST & DISPLAY OUTPUT (Fixed Array Scalar Error)
# =====================================================================
st.markdown("---")
if st.button("🚀 Calculate Future Demand Forecast", type="primary"):
    # Generate prediction array
    predicted_log = model.predict(input_data)
    predicted_units = np.expm1(predicted_log)
    
    # FIX: Safely pull index [0] to make it a Python scalar number
    final_units = int(np.round(predicted_units[0]))
    estimated_revenue = predicted_units[0] * unit_price
    
    st.balloons()
    st.success("### Forecast Calculation Successful!")
    
    # Informative subtitle identifying the product being reported
    st.markdown(f"**Target Item:** `{input_stock_code}` — *{input_description.upper()}*")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label=f"Predicted Demand Volume", value=f"{final_units} units")
    with metric_col2:
        st.metric(label="Estimated Line Revenue ($)", value=f"${estimated_revenue:,.2f}")
