import streamlit as st
import numpy as np
import gc
import pandas as pd
from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


# ------------------------------ ---------------------#
# Combined Handling for Missing Values and Duplicates
# ------------------------------ ---------------------#
def handle_missing_values(data):
    """Handles missing values and duplicates based on user selection."""
    st.header("2. ✨ Data Cleaning")
    st.subheader("🧹 Missing Value & Duplicate Handling")
    original_rows = data.shape[0]
    original_missing = data.isnull().sum().sum()
    original_duplicates = data.duplicated().sum()

    if original_missing == 0 and original_duplicates == 0:
        st.success("✅ No missing values or duplicate rows detected!")
        return data

    st.metric("Total Missing Values Found", original_missing)
    st.metric("Total Duplicate Rows Found", original_duplicates)

    # Option to treat empty strings as missing
    handle_empty_strings = st.checkbox("Treat empty strings ('') as missing values?", value=False)

    cleaning_enabled = st.checkbox("Enable Data Cleaning Options?",
                                   value=(original_missing > 0 or original_duplicates > 0 or handle_empty_strings))

    clean_data = data.copy()

    if cleaning_enabled:
        if handle_empty_strings:
            st.write("Replacing empty strings ('') with NaN...")
            try:
                clean_data.replace('', np.nan, inplace=True)
                st.success("Replaced empty strings.")
                # Update missing count after replacement
                original_missing = clean_data.isnull().sum().sum()
                st.metric("Missing Values after empty string replacement", original_missing)
            except Exception as ex:
                st.error(f"Error replacing empty strings: {ex}")

        col1, col2 = st.columns(2)
        missing_option = "Leave As Is"
        duplicate_option = "Leave As Is"

        with col1:
            if original_missing > 0:
                missing_option = st.radio(
                    "Missing Value Handling Strategy:",
                    ("Impute (Median/Mode)", "Drop Rows with NA", "Leave As Is"),
                    index=2,  # Default to Leave As Is
                    key="missing_val_radio",
                    help="""
                    - **Impute:** Fill numerical with Median, categorical with Mode.
                    - **Drop Rows with NA:** Remove rows containing *any* missing value (NaN/None).
                    - **Leave As Is:** Do not modify missing values.
                     """
                )
            else:
                st.info("No missing values (NaN/None) to handle.")

        with col2:
            if original_duplicates > 0:
                duplicate_option = st.radio(
                    "Duplicate Row Handling Strategy:",
                    ("Remove Duplicates", "Leave As Is"),
                    index=1,  # Default to Leave As Is
                    key="duplicate_radio",
                    help="Remove identical rows."
                )
            else:
                st.info("No duplicate rows to handle.")

        # --- Apply Cleaning Logic ---
        summary = ["**Cleaning Summary:**"]

        # 1. Handle Missing Values
        if missing_option == "Impute (Median/Mode)":
            imputed_num = 0
            imputed_cat = 0
            try:
                num_cols = clean_data.select_dtypes(include=np.number).columns
                cat_cols = clean_data.select_dtypes(include=['object', 'category']).columns

                # Impute Numeric
                if not clean_data[num_cols].isnull().sum().sum() == 0:
                    median_imputer = SimpleImputer(strategy='median')
                    clean_data[num_cols] = median_imputer.fit_transform(clean_data[num_cols])
                    imputed_num = 1
                    summary.append("- Imputed missing numerical values using **median**.")

                # Impute Categorical
                if not clean_data[cat_cols].isnull().sum().sum() == 0:
                    mode_imputer = SimpleImputer(strategy='most_frequent')
                    clean_data[cat_cols] = mode_imputer.fit_transform(clean_data[cat_cols])
                    imputed_cat = 1
                    summary.append("- Imputed missing categorical values using **mode**.")

                if not imputed_num and not imputed_cat:
                    summary.append("- No numerical or categorical NaNs found to impute.")

            except Exception as ex:
                st.error(f"Error during imputation: {ex}")
                summary.append("- **Error during imputation.**")


        elif missing_option == "Drop Rows with NA":
            rows_before = clean_data.shape[0]
            try:
                clean_data.dropna(inplace=True)
                rows_after = clean_data.shape[0]
                if rows_before > rows_after:
                    summary.append(f"- Dropped **{rows_before - rows_after}** rows containing missing values.")
                else:
                    summary.append("- No rows needed dropping for missing values.")
            except Exception as ex:
                st.error(f"Error dropping NA rows: {ex}")
                summary.append("- **Error during dropping NA rows.**")
        else:
            summary.append("- Left missing values (NaN/None) unchanged.")

        # 2. Handle Duplicates
        if duplicate_option == "Remove Duplicates":
            rows_before = clean_data.shape[0]
            try:
                clean_data.drop_duplicates(inplace=True)
                rows_after = clean_data.shape[0]
                if rows_before > rows_after:
                    summary.append(f"- Removed **{rows_before - rows_after}** duplicate rows.")
                else:
                    summary.append("- No duplicate rows found to remove.")
            except Exception as ex:
                st.error(f"Error removing duplicates: {ex}")
                summary.append("- **Error removing duplicates.**")
        else:
            summary.append("- Left duplicate rows unchanged.")

        # --- Verification and Return ---
        final_missing = clean_data.isnull().sum().sum()
        final_duplicates = clean_data.duplicated().sum()
        final_rows = clean_data.shape[0]

        st.markdown("\n".join(summary))

        if (missing_option == "Impute (Median/Mode)" or missing_option == "Drop Rows with NA") and final_missing > 0:
            st.warning(
                f"⚠️ **Warning:** Cleaning strategy '{missing_option}' was selected, but **{final_missing}** missing values still remain. This might cause issues in later steps.")
        elif final_missing == 0 and missing_option != "Leave As Is":
            st.success("✅ Missing values successfully handled.")

        if duplicate_option == "Remove Duplicates" and final_duplicates > 0:
            st.warning(
                f"⚠️ **Warning:** 'Remove Duplicates' was selected, but **{final_duplicates}** duplicates still remain (this shouldn't happen, check logic).")
        elif final_duplicates == 0 and duplicate_option == "Remove Duplicates":
            st.success("✅ Duplicate rows successfully removed.")

        st.metric("Rows Remaining", final_rows, delta=f"{final_rows - original_rows} vs original")

        st.write("**Preview of Data after Cleaning:**")
        st.dataframe(clean_data.head())
        gc.collect()
        return clean_data

    else:
        st.info("Data cleaning options disabled. Using original data.")
        return data


def handle_missing_values_logic(data_input, treat_empty_as_na, missing_strategy, duplicate_strategy):
    clean_data = data_input.copy()
    summary_lines = []
    action_taken = False

    if treat_empty_as_na:
        try:
            cols_before_empty_replace = clean_data.isnull().sum().sum()
            clean_data.replace('', pd.NA, inplace=True)
            cols_after_empty_replace = clean_data.isnull().sum().sum()
            if cols_after_empty_replace > cols_before_empty_replace:
                summary_lines.append(f"- Treated empty strings as missing. Missing count changed from "
                                     f"{cols_before_empty_replace} to {cols_after_empty_replace}.")
                action_taken = True
            else:
                summary_lines.append("- Checked for empty strings, none found or no change in missing count.")
        except Exception as e:
            summary_lines.append(f"- Error replacing empty strings: {e}")

    current_missing_count = clean_data.isnull().sum().sum()
    if missing_strategy == "Impute (Median/Mode)" and current_missing_count > 0:
        action_taken = True
        num_cols = clean_data.select_dtypes(include=np.number).columns
        cat_cols = clean_data.select_dtypes(include=['object', 'category']).columns
        imputed_something = False
        if not clean_data[num_cols].empty and clean_data[num_cols].isnull().sum().sum() > 0:
            median_imputer = SimpleImputer(strategy='median')
            clean_data[num_cols] = median_imputer.fit_transform(clean_data[num_cols])
            summary_lines.append("- Imputed missing numerical values (median).")
            imputed_something = True
        if not clean_data[cat_cols].empty and clean_data[cat_cols].isnull().sum().sum() > 0:
            mode_imputer = SimpleImputer(strategy='most_frequent')
            clean_data[cat_cols] = mode_imputer.fit_transform(clean_data[cat_cols])
            summary_lines.append("- Imputed missing categorical values (mode).")
            imputed_something = True
        if not imputed_something:
            summary_lines.append("- Imputation selected, but no missing values found in applicable columns.")

    elif missing_strategy == "Drop Rows with NA" and current_missing_count > 0:
        action_taken = True
        rows_before_drop = clean_data.shape[0]
        clean_data.dropna(inplace=True)
        rows_after_drop = clean_data.shape[0]
        summary_lines.append(f"- Dropped {rows_before_drop - rows_after_drop} rows containing missing values.")

    elif current_missing_count > 0:  # Leave as is selected
        summary_lines.append(f"- Missing values ({current_missing_count}) left as is.")

    current_duplicate_count = clean_data.duplicated().sum()
    if duplicate_strategy == "Remove Duplicates" and current_duplicate_count > 0:
        action_taken = True
        rows_before_dedupe = clean_data.shape[0]
        clean_data.drop_duplicates(inplace=True)
        rows_after_dedupe = clean_data.shape[0]
        summary_lines.append(f"- Removed {rows_before_dedupe - rows_after_dedupe} duplicate rows.")
    elif current_duplicate_count > 0:  # Leave as is selected
        summary_lines.append(f"- Duplicate rows ({current_duplicate_count}) left as is.")

    if not summary_lines:
        summary_lines.append("- No specific cleaning actions applied based on selections or data state.")

    final_missing = clean_data.isnull().sum().sum()
    final_duplicates = clean_data.duplicated().sum()
    gc.collect()
    return clean_data, "\n".join(
        summary_lines) if summary_lines else "No cleaning actions performed.", final_missing, final_duplicates

# -------------------------------#
# Feature engineering automate
# -------------------------------#


def encoding_categorical_features(data):
    """Encodes categorical features using selected method."""
    st.header("3. ⚙️ Feature Engineering - Encoding")
    st.subheader("🔡 Encode Categorical Variables")

    cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
    encoded_data = data.copy()

    if not cat_cols:
        st.success("No categorical feature detected in the dataset . Skipping categorical encoding ")
        return data, "No categorical columns found to encode"
    st.write(f"Found {len(cat_cols)} categorical column(s): `{', '.join(cat_cols)}`")
    if "encoding_method_index" not in st.session_state:
        st.session_state.encoding_method_index = 0
    encoding_options = ["None", "One-Hot Encoding", "Label Encoding (Ordinal)", "Binary Encoding"]
    selected_index = st.selectbox(
        "Choose an Encoding method: ",
        options=range(len(encoding_options)),  # Use range for index
        format_func=lambda x: encoding_options[x],  # Display names
        index=st.session_state.encoding_method_index,  # Use stored index
        key="encoding_selectbox",  # Assign a key
        help="""
          - **None:** Keep categorical features as they are.
          - **One-Hot:** Creates new binary columns for each category. Best for nominal data with few categories.
          - **Label:** Assigns a numerical label (0, 1, 2...). Assumes order. Suitable for ordinal data.
          - **Binary:** Creates fewer columns than One-Hot, useful for high cardinality features.
          - **Frequency:** Replaces categories with their frequencies.
          """
    )
    st.session_state.encoding_method_index = selected_index
    encoding_method = encoding_options[selected_index]

    # ---- Apply encoding based on current selection ----
    try:
        if encoding_method == "None":
            st.info("No encoding applied.")
        elif encoding_method == "One-Hot Encoding":
            encoder = OneHotEncoder(cols=cat_cols, use_cat_names=True, handle_unknown='ignore', return_df=True)
            encoded_data = encoder.fit_transform(encoded_data)
            st.success(f"Applied **One-Hot Encoding** to {len(cat_cols)} feature(s).")
        elif encoding_method == "Label Encoding (Ordinal)":
            encoder = OrdinalEncoder(cols=cat_cols, handle_unknown='value', return_df=True)
            encoded_data = encoder.fit_transform(encoded_data)
            st.success(f"Applied **Label Encoding** to {len(cat_cols)} feature(s). Assumes ordinal relationship.")
        elif encoding_method == "Binary Encoding":
            encoder = BinaryEncoder(cols=cat_cols, handle_unknown='indicator', return_df=True)
            encoded_data = encoder.fit_transform(encoded_data)
            st.success(f"Applied **Binary Encoding** to {len(cat_cols)} feature(s).")

        st.write("**Data after Encoding:**")
        st.dataframe(encoded_data.head(7))
    except ImportError as ex:
        st.error(f"Import Error: {ex}. Please install the required library (e.g., `pip install category_encoders`).")
        return data, "Import Error. please install the required library (e.g., pip install category_encoders)."
    except Exception as ex:
        st.error(f"An error occurred during encoding ({encoding_method}): {ex}")
        return data, f"An error occurred during encoding ({encoding_method}): {ex}"
    # gc.collect()
    return encoded_data, f"Applied {encoding_method} to {len(cat_cols)} categorical features."


def encoding_categorical_features_logic(data_input, encoding_method_choice):
    encoded_data = data_input.copy()
    cat_cols = encoded_data.select_dtypes(include=["object", "category"]).columns.tolist()
    enc_msg = f"Encoding method: {encoding_method_choice}."

    if not cat_cols:
        return encoded_data, "No categorical feature to encode."
    if encoding_method_choice == "None":
        return encoded_data, enc_msg + "No encoding applied."

    try:
        if encoding_method_choice == "One-Hot Encoding":
            encoder = OneHotEncoder(cols=cat_cols, use_cat_names=True, handle_unknown='ignore', return_df=True)
            encoded_data = encoder.fit_transform(encoded_data)
        elif encoding_method_choice == "Label Encoding (Ordinal)":
            encoder = OrdinalEncoder(cols=cat_cols, handle_unknown='value', handle_missing='value', return_df=True)
            encoded_data = encoder.fit_transform(encoded_data)
        elif encoding_method_choice == "Binary Encoding":
            encoder = BinaryEncoder(cols=cat_cols, handle_unknown='indicator', handle_missing='indicator',
                                    return_df=True)
            encoded_data = encoder.fit_transform(encoded_data)
        enc_msg += f" Applied to {len(cat_cols)} feature(s)."
    except Exception as e:
        enc_msg += f" Error: {e}"
        return data_input, enc_msg  # it returns original data base on error
    gc.collect()
    return encoded_data, enc_msg


def detect_outliers(column):
    """Detect if a column has outliers using then IQR method"""
    if pd.api.types.is_numeric_dtype(column) and column.nunique() > 1:
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        IQR = q3 - q1
        #  Avoid division by zero or issues if IQR is 0
        if IQR > 0:
            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR
            return ((column < lower_bound) | (column > upper_bound)).sum()
    return 0


def feature_scaling(data):
    """Applies feature scaling based on user selection or auto-detection."""
    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    num_cols = [col for col in num_cols if not set(data[col].dropna().unique()).issubset({0, 1})]  # Added dropna()
    scaled_data = data.copy()
    scale_msg = "Feature scaling not yet performed."
    scaler_summary_dict = {}

    if not num_cols:
        return data, "No numeric columns found for scaling.", {}


    st.write(f"Found {len(num_cols)} numerical column(s) for potential scaling: `{', '.join(num_cols)}`")

    # --- Persist selection ---
    if 'scaling_option_index' not in st.session_state:
        st.session_state.scaling_option_index = 0  # Default to None

    scaling_options = ["None", "Auto-Detect (Recommended)", "Manual Selection (Apply to all)"]
    selected_scaling_index = st.radio(
        "Choose Feature Scaling Method:",
        options=range(len(scaling_options)),
        format_func=lambda x: scaling_options[x],
        index=st.session_state.scaling_option_index,
        key="scaling_radio",  # Assign a key
        help="""
        - **None:** Do not scale features.
        - **Auto-Detect:** Applies StandardScaler for large ranges/variance, RobustScaler if outliers (IQR method) detected, MinMaxScaler otherwise.
        - **Manual:** Apply the chosen scaler (MinMax, Standard, Robust) to *all* detected numerical columns.
        """
    )
    #updata session state
    st.session_state.scaling_option_index = selected_scaling_index
    scaling_option = scaling_options[selected_scaling_index]

    applied_scalers_summary = {}  # Track which scaler was used for which column in Auto

    try:
        if scaling_option == "None":
            scale_msg = "No feature scaling applied as per selection."
            st.info(scale_msg)
            return scaled_data, scale_msg, scaler_summary_dict

        elif scaling_option == "Auto-Detect (Recommended)":
            st.write("Applying Auto-Detect Scaling...")
            for col in num_cols:
                # Basic checks
                if scaled_data[col].nunique() <= 1:
                    applied_scalers_summary[col] = "Skipped (Constant Value)"
                    continue
                if scaled_data[col].var() < 1e-9:
                    applied_scalers_summary[col] = f"Skipped (Near-Zero Variance: {scaled_data[col].var():.2e})"
                    continue

                outlier_count = detect_outliers(scaled_data[col])
                has_outliers = outlier_count > 0
                col_range = scaled_data[col].max() - scaled_data[col].min()
                col_std_dev = scaled_data[col].std()
                scaler_instance = None
                scaler_name_desc = ""
                if has_outliers:
                    scaler_instance = RobustScaler()
                    scaler_name_desc = f"RobustScaler (Outliers: {outlier_count})"
                elif col_range > 1000 or col_std_dev > 100:  # Arbitrary threshold for large range
                    scaler_instance = StandardScaler()
                    scaler_name_desc = f"StandardScaler (StdDev: {col_std_dev:.2f}, Range: {col_range:.2f})"
                else:
                    scaler_instance = MinMaxScaler()
                    scaler_name_desc = "MinMaxScaler (Default)"

                if scaler_instance:
                    try:
                        scaled_data[[col]] = scaler_instance.fit_transform(scaled_data[[col]])
                        applied_scalers_summary[col] = scaler_name_desc
                    except ValueError as vle:
                        st.error(f"Error scaling column '{col}'  with {scaler_name_desc}: {vle}. Skipping column")
                        applied_scalers_summary[col] = f"Error ({vle})"
                    except Exception as ex:
                        st.error(f"Error scaling column `{col}` with {scaler_name_desc}: {ex}")
                        applied_scalers_summary[col] = f"Error ({ex})"

            scale_msg = "Auto-detect scaling applied."
            scaler_summary_dict = applied_scalers_summary
            st.success(f"✅ {scale_msg}")
            # Return 3 values
            return scaled_data, scale_msg, scaler_summary_dict
        elif scaling_option == "Manual Selection (Apply to all)":
            if 'manual_scaler_type_index' not in st.session_state:
                st.session_state.manual_scaler_type_index = 0
            manual_scaler_options = ["MinMaxScaler", "StandardScaler", "RobustScaler"]
            selected_manual_index = st.selectbox(
                "Select Scaler Type for all numerical columns:",
                options=range(len(manual_scaler_options)),
                format_func=lambda x: manual_scaler_options[x],
                index=st.session_state.manual_scaler_type_index,
                key="manual_scaler_selectbox_fs"  # Changed key
            )
            st.session_state.manual_scaler_type_index = selected_manual_index
            scaler_type = manual_scaler_options[selected_manual_index]

            scaler_instance_manual = None
            if scaler_type == "MinMaxScaler":
                scaler_instance_manual = MinMaxScaler()
            elif scaler_type == "StandardScaler":
                scaler_instance_manual = StandardScaler()
            else:
                scaler_instance_manual = RobustScaler()

            valid_num_cols = [col for col in num_cols if data[col].nunique(dropna=False) > 1 and data[
                col].var() > 1e-9]  # Added dropna=False to nunique
            skipped_cols_count = len(num_cols) - len(valid_num_cols)
            if skipped_cols_count > 0:
                st.warning(f"Skipped {skipped_cols_count} columns with constant or near-zero variance.")

            if valid_num_cols:
                scaled_data[valid_num_cols] = scaler_instance_manual.fit_transform(scaled_data[valid_num_cols])
                scale_msg = f"Applied **{scaler_type}** to {len(valid_num_cols)} numerical features."
                for col in valid_num_cols: applied_scalers_summary[col] = scaler_type
                st.success(f"✅ {scale_msg}")
            else:
                scale_msg = "No valid numerical columns found to apply manual scaling."
                st.warning(scale_msg)

            scaler_summary_dict = applied_scalers_summary
        if scaling_option != 'None':
            st.write("**Data after Scaling:**")
            st.dataframe(scaled_data.head(10))
            if scaler_summary_dict:
                with st.expander("Show scaling Details"):
                    st.json(scaler_summary_dict)
        # Ensure 3 values are returned from all paths
        return scaled_data, scale_msg, scaler_summary_dict

    except ImportError as ex:
        scale_msg = f"Import Error: {ex}. Please install scikit-learn (`pip install scikit-learn`)."
        st.error(scale_msg)
        return data, scale_msg, {}
    except Exception as ex:
        scale_msg = f"An error occurred during scaling ({scaling_option}): {ex}"
        st.error(scale_msg)
        st.write("**Data (potentially unscaled or partially scaled) due to error:**")
        st.dataframe(data.head(10))  # Show original data's head in case of major error
        return data, scale_msg, {}  # Return original data on error
    gc.collect()
    return scaled_data, scale_msg, scaler_summary_dic


def feature_scaling_logic(data_input, scaling_option_choice, manual_scaler_type_choice=None):
    scaled_data = data_input.copy()
    num_cols = scaled_data.select_dtypes(include=np.number).columns.tolist()
    potential_num_cols = [col for col in num_cols if
                          scaled_data[col].nunique() > 1 and not set(scaled_data[col].dropna().unique()).issubset({0, 1})
                          ]
    applied_scalers_summary = {}
    scale_msg = f"Scaling option: {scaling_option_choice}."

    if not potential_num_cols:
        return scaled_data, "No numerical features to scale.", {}
    if scaling_option_choice == "None":
        return scaled_data, scale_msg + " No scaling applied.", {}

    try:
        if scaling_option_choice == "Auto-Detect (Recommended)":
            for col in num_cols:
                # Basic checks
                if scaled_data[col].nunique() <= 1:
                    applied_scalers_summary[col] = "Skipped (Constant Value)"
                    continue
                if scaled_data[col].var() < 1e-9:
                    applied_scalers_summary[col] = f"Skipped (Near-Zero Variance: {scaled_data[col].var():.2e})"
                    continue

                outlier_count = detect_outliers(scaled_data[col])
                has_outliers = outlier_count > 0
                col_range = scaled_data[col].max() - scaled_data[col].min()
                col_std_dev = scaled_data[col].std()
                scaler_instance = None
                scaler_name_desc = ""
                if has_outliers:
                    scaler_instance = RobustScaler()
                    scaler_name_desc = f"RobustScaler (Outliers: {outlier_count})"
                elif col_range > 1000 or col_std_dev > 100:  # Arbitrary threshold for large range
                    scaler_instance = StandardScaler()
                    scaler_name_desc = f"StandardScaler (StdDev: {col_std_dev:.2f}, Range: {col_range:.2f})"
                else:
                    scaler_instance = MinMaxScaler()
                    scaler_name_desc = "MinMaxScaler (Default)"

                if scaler_instance:
                    try:
                        scaled_data[[col]] = scaler_instance.fit_transform(scaled_data[[col]])
                        applied_scalers_summary[col] = scaler_name_desc
                    except ValueError as vle:
                        applied_scalers_summary[col] = f"Error ({vle})"
                    except Exception as ex:
                        applied_scalers_summary[col] = f"Error ({ex})"

            scale_msg = "Auto-detect scaling applied."
            scaler_summary_dict = applied_scalers_summary

        elif scaling_option_choice == "Manual Selection (Apply to all)":
            if manual_scaler_type_choice == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif manual_scaler_type_choice =="StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            scaled_data[potential_num_cols] = scaler.fit_transform(scaled_data[potential_num_cols])
            for col in potential_num_cols:
                applied_scalers_summary[col] = manual_scaler_type_choice
            scale_msg += f" Applied {manual_scaler_type_choice}."
    except Exception as e:
        scale_msg += f" Error: {e}"
        return data_input, scale_msg, {}
    gc.collect()
    return scaled_data, scale_msg, applied_scalers_summary
