import streamlit as st
import pandas as pd
from model_utils import load_data, get_model, forecast_sku
import os
import tensorflow as tf

if "page" not in st.session_state:
    st.session_state.page = 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

def next_page(total):
    if st.session_state.page < total:
        st.session_state.page += 1

st.set_page_config(
    page_title="SKU Factual Accuracy", 
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
    
    /* Reduce top empty space for main body */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    

    
    /* Hide the chain link icons next to headers */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }
    /* Hide Streamlit Cloud Default UI Elements */
    header {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* Hide the entire native dataframe hover toolbar (Download, Search, Fullscreen) */
    div[data-testid="stElementToolbar"] {
        display: none !important;
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
    st.markdown("""
        <div style='margin-bottom: 1rem;'>
            <h2 style='margin-bottom: 0rem; padding-bottom: 0rem;'>Data & Settings</h2>
            <p style='margin-top: 0.25rem;'>Upload your dataset to begin forecasting.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload SKU Dataset (CSV)", type="csv")
    
    st.divider()
    
    st.markdown("### Model Configuration")
    force_retrain = st.toggle("Force Retrain Model", value=False, help="Enable this to retrain the model from scratch instead of loading a cached version.")
    
    st.divider()
    

# --- MAIN CONTENT ---
st.markdown("<h2>📈 SKU Manufacture & Factual Accuracy Prediction </h2>", unsafe_allow_html=True)
st.markdown("### Generate high-precision for future manufacturing forecasts.")

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
    with st.expander("📊 View Dataset Summary"):
        overall_accuracy = (df["Factual Qty (Sold)"].sum() / df["Actual Qty (Produced)"].sum()) * 100
        st.success(f"Successfully loaded {len(skus)} unique SKUs ({len(df)} total records).")
        st.info(f"🎯 **Overall Historical Accuracy of Dataset:** {overall_accuracy:.2f}%")
        
        if "page" not in st.session_state:
            st.session_state.page = 1
            
        search_query = st.text_input("🔍 Search Dataset", placeholder="Type SKU, Month, or number to filter...", key="table_search")
        
        display_df = df.copy()
        display_df["Month_dt"] = display_df["Month_dt"].dt.strftime('%Y-%m-%d')
        if search_query:
            mask = display_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False, regex=False).any(), axis=1)
            display_df = display_df[mask]
            
        table_container = st.container()
        
        pag_col1, pag_col2, pag_col3, pag_col4, _ = st.columns([1.5, 0.75, 0.75, 1.5, 2])
        with pag_col1:
            rows_per_page = st.selectbox("Rows per page", [5, 10, 20, 50], index=1, key="rows_per_page")
            
        total_pages = max(1, (len(display_df) - 1) // rows_per_page + 1)
        if st.session_state.page > total_pages:
            st.session_state.page = max(1, total_pages)

        with pag_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("◀", on_click=prev_page, disabled=(st.session_state.page <= 1), use_container_width=True, key="prev_btn")
                
        with pag_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("▶", on_click=next_page, args=(total_pages,), disabled=(st.session_state.page >= total_pages), use_container_width=True, key="next_btn")
                
        with pag_col4:
            st.markdown("<br><div style='padding-top: 10px;'>Page <b>{}</b> of {}</div>".format(st.session_state.page, total_pages), unsafe_allow_html=True)
            
        start_idx = (st.session_state.page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        with table_container:
            if len(display_df) == 0:
                st.warning("No matches found for your search.")
            else:
                st.dataframe(display_df.iloc[start_idx:end_idx], width="stretch")

    st.markdown("---")
    
    # --- Forecasting Interface ---
    st.markdown("## 🔮 Generate Forecast")
    
    # Layout with columns
    with st.form("forecast_form"):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_sku = st.selectbox("Select Target SKU", skus, help="Choose the product SKU to forecast.")
            last_month = df[df["SKU"] == selected_sku]["Month_dt"].max()
            st.caption(f"Latest data available: **{last_month:%b %Y}**")
            
        with col2:
            import pandas as pd
            
            # Generate future months from Mar 2026 to Dec 2030
            future_dates = pd.date_range(start="2026-03-01", end="2030-12-01", freq="MS")
            future_months = [d.strftime("%b-%Y") for d in future_dates]
            
            target_month_str = st.selectbox(
                "Target Future Month",
                options=future_months,
                help="Select your target month and year."
            )
            st.caption(f"Selected: **{target_month_str}**")
            
        with col3:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer to align button
            generate_btn = st.form_submit_button("🚀 Forecast", type="primary", use_container_width=True)
        
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
