# pages/1_🧹_CleaningData.py
import streamlit as st
import pandas as pd  # Keep pandas if you do any direct df manipulation here
import gc
# Import from your REFACTORED utility file (e.g., utils/preprocessing.py)
from utils.cleaning import handle_missing_values_logic, encoding_categorical_features_logic, feature_scaling_logic
# from eda import assess_data_quality  # Assuming assess_data_quality is in utils/eda.py

st.set_page_config(layout="wide", page_title="Preprocess Data")
st.title("🧹⚙️ Step 2: Data Cleaning & Feature Engineering")
st.write("Clean missing values, duplicates, encode categorical features, and scale numerical features.")
st.divider()

# --- Prerequisite Check ---
if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("⬅️ Please upload a dataset on the 'Home' page first.")
    st.stop()

original_dataset_for_page = st.session_state.dataset

# --- Section 1: Data Cleaning ---
with st.container():  # Use container to group UI
    st.header("✨ Data Cleaning")

    # --- UI Controls for Cleaning ---
    # Use session state to remember choices with unique keys for this page
    st.session_state.setdefault('page_clean_handle_empty', False)
    st.session_state.setdefault('page_clean_missing_idx', 2)  # Default to "Leave As Is"
    st.session_state.setdefault('page_clean_duplicate_idx', 1)  # Default to "Leave As Is"

    handle_empty_option_ui = st.checkbox(
        "Treat empty strings ('') as missing values?",
        value=st.session_state.page_clean_handle_empty,
        key="page_clean_handle_empty_cb"  # Key to update state
    )
    # Persist the choice
    if handle_empty_option_ui != st.session_state.page_clean_handle_empty:
        st.session_state.page_clean_handle_empty = handle_empty_option_ui
        # When options change, clear results to force re-calculation
        st.session_state.cleaned_data = None
        st.session_state.processed_data = None
        st.rerun()

    # Calculate current missing/duplicates based on original_dataset + empty string choice for UI feedback
    temp_data_for_ui_check = original_dataset_for_page.copy()
    if st.session_state.page_clean_handle_empty:  # Use persisted choice
        temp_data_for_ui_check.replace('', pd.NA, inplace=True)

    current_missing_count_ui = temp_data_for_ui_check.isnull().sum().sum()
    current_duplicate_count_ui = original_dataset_for_page.duplicated().sum()

    missing_options_ui = ["Impute (Median/Mode)", "Drop Rows with NA", "Leave As Is"]
    duplicate_options_ui = ["Remove Duplicates", "Leave As Is"]

    col_clean1, col_clean2 = st.columns(2)
    with col_clean1:
        st.subheader("Missing Values")
        st.caption(f"Potential missing (NaN/None/Empty if checked): {current_missing_count_ui}")
        if current_missing_count_ui > 0:
            missing_idx_ui = st.radio(
                "Strategy:", options=range(len(missing_options_ui)),
                format_func=lambda x: missing_options_ui[x],
                index=st.session_state.page_clean_missing_idx,
                key="page_clean_missing_radio"
            )
            if missing_idx_ui != st.session_state.page_clean_missing_idx:
                st.session_state.page_clean_missing_idx = missing_idx_ui
                st.session_state.cleaned_data = None
                st.session_state.processed_data = None
                st.rerun()
            chosen_missing_strategy_ui = missing_options_ui[st.session_state.page_clean_missing_idx]

        else:
            st.info("No missing values (NaN/None/Empty) detected to handle with specific strategies.")
            chosen_missing_strategy_ui = "Leave As Is"
    with col_clean2:
        st.subheader("Duplicate Rows")
        st.caption(f"Current duplicate rows: {current_duplicate_count_ui}")
        if current_duplicate_count_ui > 0:
            duplicate_idx_ui = st.radio(
                "Strategy:", options=range(len(duplicate_options_ui)),
                format_func=lambda x: duplicate_options_ui[x],
                index=st.session_state.page_clean_duplicate_idx,
                key="page_clean_duplicate_radio"
            )
            if duplicate_idx_ui != st.session_state.page_clean_duplicate_idx:
                st.session_state.page_clean_duplicate_idx = duplicate_idx_ui
                st.session_state.cleaned_data = None
                st.session_state.processed_data = None
                st.rerun()
            chosen_duplicate_strategy_ui = duplicate_options_ui[st.session_state.page_clean_duplicate_idx]
        else:
            st.info("No duplicate rows detected.")
            chosen_duplicate_strategy_ui = "Leave As Is"

    if st.button("🧹 Apply Cleaning Steps", key="page_apply_cleaning_btn"):
        with st.spinner("Applying data cleaning..."):
            cleaned_df_result, summary_msg_result, final_missing_res, final_dupes_res = handle_missing_values_logic(
                original_dataset_for_page,  # Always pass original dataset from app.py
                st.session_state.page_clean_handle_empty,
                chosen_missing_strategy_ui,
                chosen_duplicate_strategy_ui
            )
            st.session_state.cleaned_data = cleaned_df_result
            st.session_state.cleaning_summary_display = summary_msg_result  # For display
            st.session_state.cleaning_missing_display = final_missing_res
            st.session_state.cleaning_dupes_display = final_dupes_res
            # Clear downstream FE data if cleaning is re-applied
            st.session_state.processed_data = None
            st.rerun()

    # Display Cleaning Results
    if st.session_state.get('cleaned_data') is not None:
        st.markdown("---")
        st.subheader("🧼 Cleaning Results")
        if st.session_state.get('cleaning_summary_display'):
            st.markdown(st.session_state.cleaning_summary_display)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Missing Values After", st.session_state.get('cleaning_missing_display', 'N/A'))
        with res_col2:
            st.metric("Duplicate Rows After", st.session_state.get('cleaning_dupes_display', 'N/A'))
        with res_col3:
            st.metric("Rows Remaining", st.session_state.cleaned_data.shape[0],
                      delta=f"{st.session_state.cleaned_data.shape[0] - original_dataset_for_page.shape[0]} rows")

        st.write("**Preview of Cleaned Data:**")
        st.dataframe(st.session_state.cleaned_data.head())
        st.success("Data cleaning process complete for current settings.")
    else:
        st.info("Configure and apply cleaning steps above.")

st.divider()

# --- Section 2: Feature Engineering ---
if st.session_state.get('cleaned_data') is not None:  # Only show if cleaning produced data
    with st.container():
        st.header("⚙️ Feature Engineering")
        data_for_fe_page = st.session_state.cleaned_data  # Input for FE is cleaned_data

        # --- UI for Encoding ---
        st.subheader("🔡 Encode Categorical Variables")
        st.session_state.setdefault('page_fe_encoding_idx', 0)  # Default to "None"
        encoding_options_ui = ["None", "One-Hot Encoding", "Label Encoding (Ordinal)", "Binary Encoding"]

        enc_idx_ui = st.selectbox(
            "Choose Encoding method:", options=range(len(encoding_options_ui)),
            format_func=lambda x: encoding_options_ui[x],
            index=st.session_state.page_fe_encoding_idx,
            key="page_fe_encoding_select"
        )
        if enc_idx_ui != st.session_state.page_fe_encoding_idx:
            st.session_state.page_fe_encoding_idx = enc_idx_ui
            st.session_state.processed_data = None  # Reset if option changes
            st.rerun()
        chosen_encoding_method_ui = encoding_options_ui[st.session_state.page_fe_encoding_idx]

        # --- UI for Scaling ---
        st.subheader("⚖️ Scale Numerical Features")
        st.session_state.setdefault('page_fe_scaling_idx', 0)  # Default to "None"
        st.session_state.setdefault('page_fe_manual_scaler_idx', 0)  # Default to "MinMaxScaler"

        scaling_options_ui = ["None", "Auto-Detect (Recommended)", "Manual Selection (Apply to all)"]
        scale_idx_ui = st.radio(
            "Choose Scaling Method:", options=range(len(scaling_options_ui)),
            format_func=lambda x: scaling_options_ui[x],
            index=st.session_state.page_fe_scaling_idx,
            key="page_fe_scaling_radio"
        )
        if scale_idx_ui != st.session_state.page_fe_scaling_idx:
            st.session_state.page_fe_scaling_idx = scale_idx_ui
            st.session_state.processed_data = None
            st.rerun()
        chosen_scaling_option_ui = scaling_options_ui[st.session_state.page_fe_scaling_idx]

        chosen_manual_scaler_ui = None
        if chosen_scaling_option_ui == "Manual Selection (Apply to all)":
            manual_scaler_options_ui = ["MinMaxScaler", "StandardScaler", "RobustScaler"]
            manual_scale_idx_ui = st.selectbox(
                "Select Scaler Type for Manual:", options=range(len(manual_scaler_options_ui)),
                format_func=lambda x: manual_scaler_options_ui[x],
                index=st.session_state.page_fe_manual_scaler_idx,
                key="page_fe_manual_scaler_select"
            )
            if manual_scale_idx_ui != st.session_state.page_fe_manual_scaler_idx:
                st.session_state.page_fe_manual_scaler_idx = manual_scale_idx_ui
                st.session_state.processed_data = None
                st.rerun()
            chosen_manual_scaler_ui = manual_scaler_options_ui[st.session_state.page_fe_manual_scaler_idx]

        if st.button("🔩 Apply Feature Engineering", key="page_apply_fe_btn"):
            with st.spinner("Applying feature engineering..."):
                # 1. Encoding
                encoded_df_result, enc_msg_result = encoding_categorical_features_logic(
                    data_for_fe_page.copy(), chosen_encoding_method_ui
                )
                st.session_state.fe_encoding_msg = enc_msg_result  # Store message

                # 2. Scaling (on the result of encoding)
                if encoded_df_result is not None:
                    scaled_df_result, scale_msg_result, scaler_summary_res = feature_scaling_logic(
                        encoded_df_result, chosen_scaling_option_ui, chosen_manual_scaler_ui
                    )
                    st.session_state.processed_data = scaled_df_result
                    st.session_state.fe_scaling_msg = scale_msg_result
                    st.session_state.fe_scaler_summary = scaler_summary_res
                else:  # Encoding failed or returned None
                    st.session_state.processed_data = None  # Could also be data_for_fe_page if encoding was 'None' and successful
                    st.session_state.fe_scaling_msg = "Scaling skipped due to encoding result."
                    st.session_state.fe_scaler_summary = {}
                st.rerun()

        # Display FE Results
        if st.session_state.get('processed_data') is not None:  # Check if FE has produced a result
            st.markdown("---")
            st.subheader("🔩 Feature Engineering Results")
            if st.session_state.get('fe_encoding_msg'): st.info(st.session_state.fe_encoding_msg)
            if st.session_state.get('fe_scaling_msg'): st.info(st.session_state.fe_scaling_msg)

            if st.session_state.get('fe_scaler_summary'):
                with st.expander("Show Applied Scaler Details (if any)"):
                    st.json(st.session_state.fe_scaler_summary)

            st.write("**Preview of Data After Feature Engineering:**")
            st.dataframe(st.session_state.processed_data.head())
            st.success("Feature engineering process complete for current settings.")
        elif st.session_state.get('fe_encoding_msg') or st.session_state.get(
                'fe_scaling_msg'):  # If messages exist but no data
            st.info("Review messages above. Feature engineering may not have completed successfully or was skipped.")


else:  # Cleaned data not available
    st.warning("Complete Data Cleaning to proceed to Feature Engineering.")

gc.collect()
st.divider()
if st.session_state.get('processed_data') is not None:
    st.info("➡️ Proceed to **Visualization** from the sidebar.")
else:
    st.warning("Complete Feature Engineering successfully to proceed.")
