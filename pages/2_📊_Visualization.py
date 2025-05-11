# pages/3_📊_Visualization.py

import streamlit as st
# import pandas as pd
from utils.viz import univariate_visualization, bivariate_visualization, multivariate_visualization
from utils.models import target_feature


st.set_page_config(layout="wide", page_title=" Data Visualization")
st.title("📊  Data Visualization (EDA)")
st.write("Explore your processed data visually.")
st.divider()

#---checking prerequisites-----
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.warning("⬅️ Please complete Feature Engineering first (check previous steps).")
    st.stop()

#---Get Data----
processed_data = st.session_state.processed_data

#---target selection----
st.subheader("🎯 Select Target Variable (for Viz/Importance)")
selected_target_variable = target_feature(processed_data)
if st.session_state.get('selected_target') != selected_target_variable:
    st.session_state.selected_target = selected_target_variable
    st.session_state.problem_type = None

if st.session_state.selected_target:
    st.success(f"Target variable selected: `{st.session_state.selected_target}`")
else:
    st.info("No target variable selected. Some visualizations might be limited.")
st.divider()


#----visualization options---
st.subheader("Visualization controls")
col_opts1, col_opts2 = st.columns(2)
with col_opts1:
    #Default values for sampling control
    DEFAULT_SAMPLE_FLAG = False
    DEFAULT_SAMPLE_SIZE = 1000
    use_sampling = st.checkbox("Use Sampling for All Visualizations?", value=DEFAULT_SAMPLE_FLAG,
                               key="viz_sampling_cb_page")

with col_opts2:
    sample_size = DEFAULT_SAMPLE_SIZE
    if use_sampling:
        max_samples = min(50000, len(processed_data))
        default_sample = min(DEFAULT_SAMPLE_SIZE, max_samples)
        sample_size = st.number_input("Sample Size:", min_value=min(100, max_samples), max_value=max_samples,
                                      value=default_sample, step=100, key="viz_sample_size_page")

viz_type = st.radio(
    "Select Visualization Type:",
    ["Univariate", "Bivariate", "Multivariate"],
    key="viz_type_radio_page",
    horizontal=True
)
st.divider()

#----call the appropriate visualization function----
if viz_type == "Univariate":
    univariate_visualization(processed_data, use_sampling, sample_size)
elif viz_type == "Bivariate":
    bivariate_visualization(processed_data, st.session_state.selected_target, use_sampling, sample_size)
elif viz_type == "Multivariate":
    multivariate_visualization(processed_data, st.session_state.selected_target, use_sampling, sample_size)

st.divider()
st.info("➡️ Proceed to **Feature Importance** and **ML Modeling** from the sidebar.")
