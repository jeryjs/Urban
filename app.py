# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from generate_data import generate_full_dataset, CROP_PROFILES, generate_single_crop
from train_model import preprocess, load_data, DATA_FILE, IRR_MODEL_FILE, GROWTH_MODEL_FILE

st.set_page_config(page_title="AI Urban Farming Dashboard", page_icon="🌱", layout="wide")

# --- helpers to load or create dataset ---
@st.cache_data(show_spinner=False)
def load_or_create_data(path=DATA_FILE, days=60, force=False):
    if force or (not os.path.exists(path)):
        df = generate_full_dataset(days=days, seed=42)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df

# --- load models if exist ---
def load_models():
    irr_model = None
    growth_model = None
    feature_cols = None
    if os.path.exists(IRR_MODEL_FILE):
        irr_model = joblib.load(IRR_MODEL_FILE)
    if os.path.exists(GROWTH_MODEL_FILE):
        growth_model = joblib.load(GROWTH_MODEL_FILE)
    if os.path.exists("model_feature_columns.joblib"):
        feature_cols = joblib.load("model_feature_columns.joblib")
    return irr_model, growth_model, feature_cols

# --- utility to build features for a single row or dataset ---
def build_features(df, feature_cols=None):
    df2 = pd.get_dummies(df, columns=["crop", "stage"], drop_first=True)
    if feature_cols is None:
        # return whichever columns available
        return df2
    # ensure all feature_cols exist, add missing with zeros
    for c in feature_cols:
        if c not in df2.columns:
            df2[c] = 0
    X = df2[feature_cols]
    return X

# --- load data & models ---
st.sidebar.title("Settings")
days = st.sidebar.slider("Days to simulate per crop", 30, 120, 60)
if st.sidebar.button("🔄 Recreate dataset (regenerate)"):
    df = load_or_create_data(days=days, force=True)
else:
    df = load_or_create_data(days=days, force=False)

irr_model, growth_model, feature_cols = load_models()

st.sidebar.markdown("---")
st.sidebar.write("Models status:")
st.sidebar.write(f"- Irrigation model: {'Loaded' if irr_model is not None else 'Not found'}")
st.sidebar.write(f"- Growth model: {'Loaded' if growth_model is not None else 'Not found'}")
st.sidebar.markdown("---")
st.sidebar.write("If models are not available, press **Train models** below.")

if st.sidebar.button("📚 Train models now"):
    # run training (simple call) - this reads the CSV and trains
    from train_model import main as train_main
    train_main()
    st.experimental_rerun()

# top layout
st.title("🌱 AI-Optimized Urban Farming System — Full Prototype")
st.write("Simulation + ML models + Web dashboard. Uses Indian organic inputs (Panchgavya/Jeevamrut) logic and model predictions.")

# select crop and day
crop_list = list(CROP_PROFILES.keys())
crop_choice = st.selectbox("Choose crop to inspect", crop_list)
day_choice = st.slider("Select day", 0, int(df['day'].max()), 0)

# filter dataset for that crop
crop_df = df[df['crop'] == crop_choice].reset_index(drop=True)
row = crop_df.iloc[min(day_choice, len(crop_df)-1)]

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
# irrigation logic (simple fallback if model missing)
def simple_irrigation_rule(moisture, ideal):
    if moisture < ideal - 10:
        return "Irrigate Now", "red"
    elif moisture < ideal:
        return "Irrigate Soon", "orange"
    else:
        return "OK", "green"

ideal_moisture = CROP_PROFILES[crop_choice]['ideal_moisture']
ir_status, ir_color = simple_irrigation_rule(row['moisture'], ideal_moisture)

with col1:
    st.metric("💧 Moisture (%)", f"{row['moisture']}%", delta=None)
with col2:
    st.metric("🧪 Nutrients", f"{row['nutrients']}")
with col3:
    st.metric("☀️ Light Hours", f"{row['light_hours']}h")
with col4:
    st.metric("🌡️ Temperature", f"{row['temperature']}°C")

# --- AI Model Predictions (if models present)
st.markdown("### 🤖 AI Recommendations")
# prepare features
single = crop_df.iloc[[min(day_choice, len(crop_df)-1)]]
X_single = build_features(single, feature_cols)

if irr_model is not None and feature_cols is not None:
    try:
        irr_pred = irr_model.predict(X_single)[0]
        proba = irr_model.predict_proba(X_single)[0]
        st.write(f"**Irrigation prediction (model):** `{irr_pred}` (probs: {np.round(proba,2)})")
    except Exception as e:
        st.warning("Model prediction failed; falling back to rule: " + str(e))
        st.write(f"**Irrigation prediction (rule):** {ir_status}")
else:
    st.write(f"**Irrigation prediction (rule):** {ir_status}")

if growth_model is not None and feature_cols is not None:
    try:
        growth_pred = growth_model.predict(X_single)[0]
        st.write(f"**Growth index prediction (model):** {growth_pred:.1f} / 100")
    except Exception as e:
        st.warning("Growth model prediction failed; falling back to simulated value.")
        st.write(f"**Growth index (simulated):** {row['growth_index']:.1f}")
else:
    st.write(f"**Growth index (simulated):** {row['growth_index']:.1f}")

# Organic inputs logic
st.markdown("### 🌿 Organic Inputs & Crop Management")
day = int(row['day'])
panch = "Add Panchgavya Today" if day % 7 == 0 else "No Panchgavya"
jeev = "Add Jeevamrut Today" if day % 10 == 0 else "No Jeevamrut"
rotation = "Rotate Crop Batch" if day % 15 == 0 else "Continue same crop"
st.write(f"- **Panchgavya**: {panch}")
st.write(f"- **Jeevamrut**: {jeev}")
st.write(f"- **Crop rotation**: {rotation}")

# --- charts
st.markdown("### 📊 Charts (30-day view)")
chart_df = crop_df.copy()
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].plot(chart_df['day'], chart_df['moisture'], marker='o')
ax[0].axhline(CROP_PROFILES[crop_choice]['ideal_moisture'], color='green', linestyle='--', alpha=0.6)
ax[0].set_title("Moisture (%)")
ax[0].set_xlabel("Day")

ax[1].plot(chart_df['day'], chart_df['nutrients'], marker='o', color='orange')
ax[1].set_title("Nutrients")
ax[1].set_xlabel("Day")

ax[2].plot(chart_df['day'], chart_df['growth_index'], marker='o', color='purple')
ax[2].set_title("Growth Index")
ax[2].set_xlabel("Day")

st.pyplot(fig)

# --- dataset preview
with st.expander("Show simulated dataset (first 50 rows)"):
    st.dataframe(df.head(50))

st.success("Dashboard ready. Tip: Use sidebar to re-simulate data or train models.")
