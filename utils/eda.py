"""
this file contain functionality like data quality assessment,
feature importance and problem type determination
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_selection import f_regression, mutual_info_regression, f_classif, mutual_info_classif
# import gc
from sklearn.preprocessing import LabelEncoder


# ------------------------------ #
#     Data Quality Assessment    #
# ------------------------------ #
def assess_data_quality(data):
    st.write(f"Dataset shape: **{data.shape[0]} rows x {data.shape[1]} columns **")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Column Data Types")
        st.dataframe(data.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'DataType'}))

    # for missing values
    with col2:
        st.subheader(" 🚨 Missing Values")
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        missing_df = pd.DataFrame({
            "Missing Values": missing_values,
            "Percentage": missing_percentage.round(2)
        })
        st.dataframe(missing_df[missing_df["Missing Values"] > 0])

    col3, col4 = st.columns(2)
    # check for duplicate values
    with col3:
        st.subheader("📍 Duplicate Rows")
        duplicate_count = data.duplicated().sum()
        st.metric("Total Duplicate Rows", duplicate_count)

    # 5️⃣ Unique values per column (for categorical insights)
    with col4:
        st.subheader("🎯 Unique Values per Column")
        unique_values = data.nunique()
        st.dataframe(unique_values.reset_index().rename(columns={'index': 'Column', 0: 'UniqueCount'}))

    # 6️⃣ Basic statistics (Numerical Columns Only)
    st.subheader("📊 Basic Statistics (Numerical Data)")
    st.dataframe(data.describe(include=np.number))
    st.subheader("📊 Basic Statistics (Categorical Data)")
    cat_cols = data.select_dtypes(include="object").columns
    if cat_cols.size > 0:
        st.write("Categorical stats:")
        st.dataframe(data.describe(include=["object"]))
    else:
        st.success("No categorical columns found for statistics .")


# function to find the type of problem that the dataset is having ( classification or regression)
# ============================================
# Problem Type Determination
# ============================================
def determine_problem_type(database, target_column):
    """
       Analyzes the target column to determine the ML problem type.
       Returns a tuple: (problem_type_string, message_string)
    """
    if database is None or database.empty:
        return "Error: No data", "Error: Dataset is not available for problem type determination."
    if target_column is None or target_column == "None":
        return None, "Info: No target variable selected for problem type determination."
    if target_column not in database.columns:
        return "Error: Target not found", f"Error: Target column '{target_column}' not found in the dataset."


    target_series = database[target_column]
    #checking for the missing values in the target column which stop further processing
    if target_series.isnull().any():
        return "Error: Target has NaNs", (f"Error: Target column '{target_column}' contains missing values. "
                                          f"Please handle them first.")

    # analyze data type and unique values
    col_dtype = target_series.dtype
    n_unique = target_series.nunique()
    n_rows = len(target_series)

    # constant target check
    if n_unique <= 1:
        return "Constant Target", (f"Warning: Target '{target_column}' is constant (all values are the same: "
                                   f"{target_series.unique()}). Cannot be used for meaningful modeling.")
    # Boolean type
    if pd.api.types.is_bool_dtype(col_dtype):
        return "Binary Classification", (f"Inferred Problem Type: Binary Classification "
                                         f"(Boolean Target: '{target_column}').")

    # object or categorical type:
    if pd.api.types.is_object_dtype(col_dtype) or isinstance(col_dtype, pd.CategoricalDtype):
        if n_unique == 2:
            return "Binary Classification", (f"Inferred Problem Type: Binary Classification (Categorical Target"
                                             f" '{target_column}' with 2 unique values: {target_series.unique()[:2]}).")
        else:
            return "Multiclass Classification", (f"Inferred Problem Type: Multiclass Classification (Categorical Target "
                                                 f"'{target_column}' with {n_unique} unique values).")
    # Numerical type (more complex logic)
    if pd.api.types.is_numeric_dtype(col_dtype):
        unique_values = target_series.unique()
        # Check for exactly two unique numeric values (common for 0/1 encoded targets)
        if n_unique == 2:
            return "Binary Classification", (f"Inferred Problem Type: Binary Classification (Numerical Target "
                                             f"'{target_column}' with 2 unique values: {unique_values[:2]}).")

        # Heuristic for integer types: if very few unique values, likely classification
        if pd.api.types.is_integer_dtype(col_dtype):
            ratio_unique_to_rows = (n_unique / n_rows) if n_rows > 0 else 0
            # Thresholds can be tuned. Example: less than 15 unique int values OR unique values are less than 5% of
            # rows (and not excessively high raw count)
            IS_LIKELY_CLASSIFICATION_INT = (n_unique < 15) or (ratio_unique_to_rows < 0.05 and n_unique < 0.1 * n_rows)
            if IS_LIKELY_CLASSIFICATION_INT:
                return "Multiclass Classification", (f"Inferred Problem Type: Multiclass Classification (Integer Target "
                                                     f"'{target_column}' with {n_unique} unique values, appears discrete).")
            else:
                return "Regression", (f"Inferred Problem Type: Regression (Integer Target '{target_column}' with "
                                      f"{n_unique} unique values, appears continuous or high cardinality).")

        # Float type (generally regression, unless very few unique values mimicking categories, but less common for floats)
        if pd.api.types.is_float_dtype(col_dtype):
            # If a float column has very few unique values (e.g., 1.0, 2.0, 3.0), it might be miscoded.
            # For a robust system, floats are usually treated as regression unless other strong indicators.
            if n_unique <= 5 and np.all(np.equal(np.mod(unique_values, 1), 0)):
                return "Multiclass Classification", (f"Warning: Target '{target_column}' is float but has only "
                                                     f"{n_unique} whole number unique values ({unique_values[:5]}). "
                                                     f"Interpreted as Multiclass Classification. Verify if this is "
                                                     f"correct or if it should be Regression.")

            return "Regression", f"Inferred Problem Type: Regression (Float Target '{target_column}')."
    return "Error: Unknown Type", (f"Error: Could not determine problem type for target '{target_column}' (dtype: "
                                   f"{col_dtype}, Unique values: {n_unique}). Please check the target column's characteristics.")


# -----------------------------
# Feature Importance Analysis
# -----------------------------
def feature_importance_analysis(data, target_column, problem_type):
    """Calculates feature importance scores. Returns DataFrame."""
    if target_column not in data.columns:
        return None, "Error: Target not found."
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_numeric = X.select_dtypes(include=np.number)
    if X_numeric.empty:
        return None, "Warn: No numerical features found."
    if X_numeric.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        return None, "Error: NaNs detected."

    results = {'Feature': X_numeric.columns}
    calculation_errors = []
    if not problem_type or "Error" in problem_type or problem_type =="Constant Target":
        return None, f"Warn: Cannot calculate scores, Problem Type: {problem_type}"

    try:
        y_for_corr = y
        if "Classification" in problem_type:
            le = LabelEncoder()
            y_encoder = le.fit_transform(y)
            y_for_corr = y_encoder
            try:
                f_scores, _ = f_classif(X_numeric, y_encoder)
                results['F-Score (Classification)'] = f_scores
            except Exception as ex:
                calculation_errors.append(f"F-Score failed: {ex}")
                results['F-Score (Classification)'] = np.nan
            try:
                mi_scores = mutual_info_classif(X_numeric, y_encoder, random_state=42)
                results["Mutual_information (Classification)"] = mi_scores
            except Exception as ex:
                calculation_errors.append(f"MI failed: {ex}")
                results["Mutual_information (Classification)"] = np.nan

        elif problem_type == "Regression":
            try:
                f_scores, _ = f_regression(X_numeric, y)
                results['F-Score (Regression)'] = f_scores
            except Exception as ex:
                calculation_errors.append(f"F-Score failed: {ex}")
                results['F-Score (Regression)'] = np.nan
            try:
                mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
                results['Mutual Information (Regression)'] = mi_scores
            except Exception as ex:
                calculation_errors.append(f"MI failed: {ex}")
                results['Mutual Information (Regression)'] = np.nan

        # pearson
        try:
            results['Pearson Correlation (abs)'] = X_numeric.corrwith(pd.Series(y_for_corr)).abs()
        except Exception as e:
            calculation_errors.append(f"Pearson failed: {e}")
            results['Pearson Correlation (abs)'] = np.nan
    except Exception as e:
        return None, f"Error:Unexpected error during score calculation: {e}"

    importance_df = pd.DataFrame(results)
    # sorting logic
    tools = ['Mutual Information (Classification)', 'Mutual Information (Regression)',
             'F-Score (Classification)', 'F-Score (Regression)', 'Pearson Correlation (abs)']
    sort_col = next((col for col in tools if col in importance_df.columns), None)
    if sort_col:
        importance_df = importance_df.sort_values(by=sort_col, ascending=False, na_position='last').reset_index(drop=True)

    return importance_df, calculation_errors
