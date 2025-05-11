# pages/0_🔍_Data_Quality.py
import streamlit as st
from utils.eda import assess_data_quality # Assuming you move assess_data_quality to utils/eda.py

st.set_page_config(layout="wide", page_title="Data Quality Assessment")
st.title("🔍 Step 1: Data Quality Assessment")
st.write("Review the initial quality and structure of your uploaded dataset.")
st.divider()

if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("⬅️ Please upload a dataset on the 'Home' page first.")
    st.stop()

dataset_to_assess = st.session_state.dataset

st.header("📋 Initial Data Overview")
assess_data_quality(dataset_to_assess) # This function already uses st commands to display info

st.divider()
st.info("➡️ Proceed to **Data Cleaning** from the sidebar.")