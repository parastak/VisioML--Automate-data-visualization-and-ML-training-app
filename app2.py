# app.py file

import streamlit as st
# import pandas as pd
# import gc
from utils.load_utils import load_dataset

# page config

st.set_page_config(
    page_title="VisioML: autoML workflow App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


#---optional : Load CSS file that we create ----
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass
# load_css("Style_css") uncomment this, when I create a style file for this app


# --- Initialize Session State ---
# This ensures keys exist when pages try to access them
# Data states - reset when a new file is uploaded

st.session_state.setdefault('dataset', None)
st.session_state.setdefault('loaded_file_id', None)
st.session_state.setdefault('cleaned_data', None)
st.session_state.setdefault('processed_data', None)
st.session_state.setdefault('selected_target', None)
st.session_state.setdefault('problem_type', None)
st.session_state.setdefault('features_to_use', None)
st.session_state.setdefault('X_train', None)
st.session_state.setdefault('X_test', None)
st.session_state.setdefault('y_train', None)
st.session_state.setdefault('y_test', None)
st.session_state.setdefault('trained_models', None)
st.session_state.setdefault('evaluation_results', None)
st.session_state.setdefault('cv_results', None)

# Widget state keys
st.session_state.setdefault('encoding_method_index', 0)
st.session_state.setdefault('scaling_option_index', 0)
st.session_state.setdefault('manual_scaler_type_index', 0)
st.session_state.setdefault('target_column_index', 0)
st.session_state.setdefault('feature_selector_ms_default', [])

#----Main App Title-----
st.title(" 🚀 VisioML:Automated Data Analysis & ML Prep App ")
st.write("Welcome ! Upload your dataset using the sidebar to begin your automated workflow journey.")
st.markdown("Use the **sidebar** (click `>` if hidden) to navigate through the analysis steps")
st.divider()

#---File upload section ui (in sidebar) ----
with st.sidebar:
    st.header("📁 Step 1: Load Data")
    upload_file = st.file_uploader(
        label="Upload your CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        key="main_app_file_uploader",
        help="Upload your dataset. Processing will begin based on your selections in the workflow pages."
    )

    if upload_file is not None:
        # checking if it's a new file to trigger reset for session state
        if st.session_state.get("loaded_file_id") != upload_file.file_id:
            st.session_state.loaded_file_id = upload_file.file_id  # setting new id first for new file
            st.info(f"✨ New file detected: '{upload_file.name}'. Processing...")
            with st.spinner("Loading data and initializing..."):
                raw_data, load_msg, _ = load_dataset(upload_file)
                if raw_data is not None and not (isinstance(load_msg, str) and load_msg.startswith("Error")):
                    st.success("Dataset Loaded in sidebar!")
                    if isinstance(load_msg, str) and load_msg.startswith("Warn"):
                        st.warning(load_msg)
                    else:
                        st.caption(load_msg)
                    raw_data.columns = raw_data.columns.str.strip().str.replace(r'[^\w\s]+', '', regex=True).str.replace(' ', '_').str.lower()
                    st.session_state.dataset = raw_data

                    # this part is critical help to reset all the session state for the new file uploaded----
                    keys_to_reset = [
                        'cleaned_data', 'processed_data', 'selected_target', 'problem_type',
                        'features_to_use', 'X_train', 'X_test', 'y_train', 'y_test',
                        'trained_models', 'evaluation_results', 'cv_results',
                        'encoding_method_index', 'scaling_option_index', 'manual_scaler_type_index',
                        'target_column_index', 'feature_selector_ms_default'
                    ]
                    for key_to_reset in keys_to_reset:
                        if key_to_reset.endswith('_index'):
                            st.session_state[key_to_reset] = 0
                        else:
                            st.session_state[key_to_reset] = None

                    # re-initializing features_to_use with new columns if dataset loaded
                    if st.session_state.dataset is not None:
                        all_cols_temp = st.session_state.dataset.columns.tolist()
                        st.session_state.features_to_use = all_cols_temp
                        st.session_state.feature_selector_ms_default = all_cols_temp

                    for key in list(st.session_state.keys()):
                        if key.startswith(('univar_', 'bivar_', 'multivar_', 'heatmap_', 'plot_model_select')):
                            del st.session_state[key]

                    st.success("🔄 Application state reset for new file.")
                else:
                    st.error(load_msg if isinstance(load_msg, str) else "Failed to load data.")
                    st.session_state.dataset = None
                    st.session_state.loaded_file_id = None

    elif upload_file is None and st.session_state.get('loaded_file_id') is not None:
        # file removed by user from uploader
        st.info("⏳ No file detected. Resetting application state...")
        keys_to_reset_on_remove = [
            'dataset', 'cleaned_data', 'processed_data', 'selected_target', 'problem_type',
            'features_to_use', 'X_train', 'X_test', 'y_train', 'y_test',
            'trained_models', 'evaluation_results', 'cv_results', 'loaded_file_id'
        ]
        for key_to_reset in keys_to_reset_on_remove:
            if key_to_reset in st.session_state:
                st.session_state[key_to_reset] = None
        widget_index_keys = ['encoding_method_index', 'scaling_option_index', 'manual_scaler_type_index', 'target_column_index']
        for key_index in widget_index_keys:
            st.session_state[key_index] = 0
        st.session_state.feature_selector_ms_default = []
        st.rerun()

# Main UI area of app.py (landing page content)
if st.session_state.get('dataset') is None:
    st.markdown("### Get Started")
    st.markdown("1. **Upload Your Data**: Use the sidebar to upload your CSV or Excel file.")
    st.markdown(
        "2. **Navigate Workflow**: Once data is loaded, pages for Cleaning, Visualization, and Modeling will become "
        "active in the sidebar.")
    st.markdown(
        "3. **Follow the Steps**: Each page guides you through a part of the data analysis and ML preparation pipeline.")

else:
    st.success(
        f"Dataset **'{upload_file.name if upload_file else st.session_state.loaded_file_id}'** is loaded and ready!")
    st.markdown("Please use the sidebar navigation to proceed through the workflow steps:")
    st.markdown("- **Data Quality**: Initial overview of your data.")
    st.markdown("- **Data Cleaning**: Handle missing values and duplicates.")
    st.markdown("- **Feature Engineering**: Encode and scale features.")
    st.markdown("- **Visualization**: Explore data with various plots.")
    st.markdown("- **Feature Importance & Selection**: Understand feature relevance and select features for modeling.")
    st.markdown("- **ML Modeling**: Split data, run cross-validation, train models, evaluate, and download.")

    st.subheader("Quick Peek at your Data: ")
    st.dataframe(st.session_state.dataset.head(15))

    # Optional part : Display data quality summary on home page
    if st.button("Show Initial Data Quality summary Here", key="home_show_qa_btn"):
        from utils.eda import assess_data_quality
        with st.expander("Data Quality Summary", expanded=True):
            assess_data_quality(st.session_state.dataset)

st.markdown("-----")
st.caption("VisioML: Automated ML Workflow App | v1.0")
