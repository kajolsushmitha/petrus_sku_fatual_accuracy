import streamlit as st
import pandas as pd
from model_utils import load_data, get_model, forecast_sku
import os
import tensorflow as tf

st.set_page_config(
    page_title="SKU Forecaster AI", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more premium look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1 {
        color: #1e3a8a;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #2563eb;
        font-weight: 600;
    }
    
    /* Custom card for results */
    .result-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-top: 4px solid #3b82f6;
        margin-top: 1rem;
    }
    
    /* Metric styling overrides */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=100)
    st.title("Data & Settings")
    st.markdown("Upload your dataset to begin forecasting.")
    
    uploaded_file = st.file_uploader("Upload SKU Dataset (CSV)", type="csv")
    
    st.divider()
    
    st.markdown("### Model Configuration")
    force_retrain = st.toggle("Force Retrain Model", value=False, help="Enable this to retrain the model from scratch instead of loading a cached version.")
    
    st.divider()
    st.caption("Powered by LSTM Neural Networks & Streamlit")

# --- MAIN CONTENT ---
st.markdown("<h1>📈 SKU Manufacture & Accuracy AI</h1>", unsafe_allow_html=True)
st.markdown("### Generate high-precision future manufacturing forecasts.")

if uploaded_file is not None:
    # Read the raw CSV
    raw_df = pd.read_csv(uploaded_file)
    
    with st.spinner("Processing data..."):
        df = load_data(raw_df)
        skus = sorted(df["SKU"].unique())
        sku_to_id = {s: i for i, s in enumerate(skus)}
    
    # We use a session state or caching to avoid reloading model constantly
    @st.cache_resource
    def load_or_train_model(df, skus, _sku_to_id, force_retrain):
        return get_model(df, skus, _sku_to_id, force_retrain=force_retrain)

    with st.spinner("🧠 Initializing Neural Network Model..."):
        model, growth_scaler, per_sku = load_or_train_model(df, skus, sku_to_id, force_retrain)
    
    # Create an expander for dataset info to keep UI clean
    with st.expander("📊 View Dataset Summary", expanded=False):
        st.success(f"Successfully loaded {len(skus)} unique SKUs.")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    
    # --- Forecasting Interface ---
    st.markdown("## 🔮 Generate Forecast")
    
    # Layout with columns
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_sku = st.selectbox("Select Target SKU", skus, help="Choose the product SKU to forecast.")
        last_month = df[df["SKU"] == selected_sku]["Month_dt"].max()
        st.caption(f"Latest data available: **{last_month:%b %Y}**")
        
    with col2:
        target_month_str = st.text_input("Target Future Month", placeholder="e.g. Jun-2026", help="Enter a month in the future using the format MMM-YYYY.")
        st.caption("Format required: **MMM-YYYY**")
        
    with col3:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer to align button
        generate_btn = st.button("🚀 Forecast", type="primary", use_container_width=True)
        
    if generate_btn:
        if not target_month_str:
            st.error("⚠️ Please enter a target month.")
        else:
            with st.spinner(f"Calculating multi-step forecast for {selected_sku}..."):
                try:
                    result = forecast_sku(
                        model=model,
                        growth_scaler=growth_scaler,
                        per_sku=per_sku,
                        df=df,
                        sku_to_id=sku_to_id,
                        sku=selected_sku,
                        target_month_str=target_month_str
                    )
                    
                    # --- Results Container ---
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Forecast Results for {selected_sku} ({result['Target Month']})")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    r_col1, r_col2, r_col3 = st.columns(3)
                    
                    # Add some styling to metrics
                    with r_col1:
                        st.metric(label="📦 Factual Qty (Forecast)", value=f"{result['Factual Qty (Forecast)']:,}")
                    
                    with r_col2:
                        st.metric(label="🎯 Predicted Accuracy", value=f"{result['Predicted Accuracy %']}%")
                        
                    with r_col3:
                        st.metric(label="⚙️ Recommended Manufacture Qty", value=f"{result['Recommended Manufacture Qty']:,}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except ValueError as e:
                    st.error(f"⚠️ {str(e)}")
                except Exception as e:
                    st.error(f"⚠️ An unexpected error occurred: {e}")
else:
    # Empty state styling
    st.markdown("""
        <div style="text-align: center; padding: 4rem; background-color: #f1f5f9; border-radius: 12px; margin-top: 2rem; border: 2px dashed #cbd5e1;">
            <h2 style="color: #64748b;">No Data Uploaded</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">Please upload your SKU dataset CSV from the sidebar to start forecasting.</p>
        </div>
    """, unsafe_allow_html=True)
