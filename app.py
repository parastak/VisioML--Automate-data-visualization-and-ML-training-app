"""
 A Streamlit-based app that automates data preprocessing, feature engineering, exploratory data analysis (EDA),
 and machine learning (ML) model training. Designed to assist data analysts and ML engineers with end-to-end workflows.

"""

#here we import all the necessary modules and libraries used in this app
import pandas as pd
import streamlit as st
# import numpy as np
import seaborn as sns
import numpy as np
# import math
import gc
import matplotlib.pyplot as plt
import joblib
import io

# encoder and scalers
# from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder
from pandas.core.dtypes.common import is_object_dtype
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder

# feature selection and importance
from sklearn.feature_selection import f_regression, mutual_info_regression, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split

# === Add these ML Model and Metrics Imports ===
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report)
# files imported
from utils.load_utils import load_dataset
from utils.cleaning import handle_missing_values, encoding_categorical_features, feature_scaling

# ==============================================

# this is the css section for the streamlit app to change design of overall app to make it more attractive and good.
# st.set_page_config(layout="wide")
# pg_bg_color = """
# <style>
# [data-testid = "stApp" ]{
#   background: rgb(223 223 223);
#   color: rgb(14 17 23);
# }
# [data-testid = "stHeader" ]{
#   background: rgb(172 173 173 / 64%);
#   display: flex;
#   align-items: center;
#   justify-content: flex-end;
#   height: 3rem;
# }
# [data-testid = "stToolbar" ]{
#   position: relative;
#   top: 0;
# }
# [data-testid = "stHorizontalBlock" ]{
#   flex-direction : column;
# }
# [data-testid = "stMainBlockContainer" ]{
#   padding: 4rem 1rem 4rem;
# }
# [data-testid = "stMarkdownContainer" ]{
#   width : 750px;
# }
# [data-testid = "stWidgetLabel" ]{
#   color: rgb(14 17 23);
# }
# .stFileUploader > div{
#   background: #bebfbf;
#   border-radius: 10px;
#   line-height: 3;
#   border: 1px solid #dfdfdf;
# }
# [data-testid = "stAlertContainer"] {
#   background-color: rgb(190 191 191);
#   color: rgb(38 39 48);
# }
# [data-testid = "stBaseButton-secondary"]{
#   width : 170px;
# }
# [data-testid = "stButton"]{
#   width: 168px;
#   color: white;
# }
# [data-testid="stColumn"]{
#   display: flex;
#   align-items:center;
# }
# </style>
# """
# st.markdown(pg_bg_color, unsafe_allow_html=True)

# -------- configuration -----------

st.title("🚀 Automated Data Analysis and ML Prep App 🤖")
st.write("Upload your dataset (CSV or Excel) to explore, clean, visualize, and analyze feature importance!")


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


# -------------------------------#
# Feature visualizing part
# -------------------------------#
def get_sampled_data(data, do_sample, n_sample_row):
    if do_sample and len(data) > n_sample_row:
        st.info(f"Using a sample of {n_sample_row} rows for visualization.")
        return data.sample(n=n_sample_row, random_state=42)
    return data


def plot_univariate(data_to_plot, col, chart_type, ax):
    """Helper to plot single univariate chart."""
    if pd.api.types.is_numeric_dtype(data_to_plot[col]):
        if chart_type == "Histogram":
            sns.histplot(data_to_plot[col], kde=True, bins=30, ax=ax)
            ax.set_title(f"Histogram - {col}")
        elif chart_type == "Box Plot":
            sns.boxplot(y=data_to_plot[col], ax=ax)
            ax.set_title(f"Box Plot - {col}")
        elif chart_type == "KDE Plot":
            sns.kdeplot(data_to_plot[col], fill=True, ax=ax)
            ax.set_title(f"KDE Plot - {col}")
        else:  # Default numerical
            sns.histplot(data_to_plot[col], kde=True, bins=30, ax=ax)
            ax.set_title(f"Histogram - {col}")
    else:  # Categorical / Object
        # Ensure Rotation for Readability
        rotation = 45 if data_to_plot[col].nunique() > 5 else 0
        order = data_to_plot[col].value_counts().index[:20]  # Limit categories shown in count/bar plots

        if chart_type == "Count Plot":
            sns.countplot(x=data_to_plot[col], ax=ax, order=order)
            ax.set_title(f"Count Plot - {col} (Top 20)")
            ax.tick_params(axis='x', rotation=rotation)
        elif chart_type == "Pie Chart":
            # Limit slices for readability
            value_counts = data_to_plot[col].value_counts()
            threshold = 0.01  # Combine slices smaller than 1%
            small_slices = value_counts[value_counts / value_counts.sum() < threshold]
            if not small_slices.empty:
                value_counts = value_counts[value_counts / value_counts.sum() >= threshold]
                value_counts['Other'] = small_slices.sum()

            if value_counts.empty:
                ax.text(0.5, 0.5, 'No data to plot', horizontalalignment='center', verticalalignment='center')
            else:
                value_counts.plot.pie(autopct="%1.1f%%", ax=ax, startangle=90, counterclock=False)
            ax.set_ylabel('')  # Remove default y-label from pie chart
            ax.set_title(f"Pie Chart - {col}")
        else:  # Default categorical
            sns.countplot(x=data_to_plot[col], ax=ax, order=order)
            ax.set_title(f"Count Plot - {col} (Top 20)")
            ax.tick_params(axis='x', rotation=rotation)


def univariate_visualization(data, do_sample, n_sample_rows):
    """Handles univariate feature visualization."""
    st.header('4. 📊 Data Visualization')
    st.subheader('📈 Univariate Analysis (Single Feature)')

    data_to_plot = get_sampled_data(data, do_sample, n_sample_rows)

    all_cols = data_to_plot.columns.tolist()
    if not all_cols:
        st.warning("No columns available for visualization.")
        return

    feature = st.selectbox("Select Feature for Univariate Analysis:", all_cols)

    if feature:
        # Determine suitable chart types
        if pd.api.types.is_numeric_dtype(data_to_plot[feature]):
            default_chart = "Histogram"
            chart_options = ["Histogram", "Box Plot", "KDE Plot"]
        else:
            default_chart = "Count Plot"
            chart_options = ["Count Plot", "Pie Chart"]

        chart_type = st.selectbox("Choose Chart Type:", chart_options, index=chart_options.index(default_chart))

        st.write(f"Generating **{chart_type}** for **{feature}**...")
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            plot_univariate(data_to_plot, feature, chart_type, ax)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"Could not generate plot for {feature}: {ex}")
        finally:
            plt.close(fig)  # Close the plot to free memory
            gc.collect()


def bivariate_visualization(data, target, do_sample, n_sample_rows):
    """Handles bivariate feature visualization."""
    st.subheader('📉 Bivariate Analysis (Two Features)')

    data_to_plot = get_sampled_data(data, do_sample, n_sample_rows)

    num_cols = data_to_plot.select_dtypes(include=np.number).columns.tolist()
    cat_cols = data_to_plot.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = num_cols + cat_cols

    if len(all_cols) < 2:
        st.warning("Need at least two columns for bivariate analysis.")
        return

    # Suggest target for Y-axis if available, otherwise default to first column
    y_index = 0
    if target and target in all_cols:
        try:
            y_index = all_cols.index(target)
        except ValueError:
            y_index = 0  # If target somehow not in list, default to 0

    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Select X-Axis Feature:", all_cols, index=0)
    with col2:
        # Default Y to target if available and different from X, else default to second column if available
        if feature_x != all_cols[y_index]:
            default_y_index = y_index
        elif len(all_cols) > 1:
            default_y_index = 1 if feature_x == all_cols[0] else 0  # Pick the other column
        else:
            default_y_index = 0  # Should not happen if len(all_cols) >= 2 check passed

        feature_y = st.selectbox("Select Y-Axis Feature:", all_cols, index=default_y_index)

    if feature_x == feature_y:
        st.warning("Please select two different features.")
        return

    st.write(f"Analyzing relationship between **{feature_x}** (X) and **{feature_y}** (Y)")

    x_is_num = feature_x in num_cols
    y_is_num = feature_y in num_cols

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_generated = False

    try:
        # Determine appropriate plot types
        if x_is_num and y_is_num:
            # Num vs Num
            plot_type = st.selectbox("Choose Plot Type:",
                                     ["Scatter Plot", "Line Plot (use with caution)", "Joint Plot"])
            if plot_type == "Scatter Plot":
                sns.scatterplot(data=data_to_plot, x=feature_x, y=feature_y, ax=ax, alpha=0.6)
                ax.set_title(f"Scatter: {feature_x} vs {feature_y}")
            elif plot_type == "Line Plot":
                # Line plots usually imply order or time, warn user
                st.warning(
                    "Line plots are best for ordered data (like time series). Ensure this makes sense for your features.")
                # Sort by X for a potentially more meaningful line
                sorted_data = data_to_plot.sort_values(by=feature_x)
                sns.lineplot(data=sorted_data, x=feature_x, y=feature_y, ax=ax, ci=None)  # ci=None faster
                ax.set_title(f"Line Plot: {feature_x} vs {feature_y}")
            elif plot_type == "Joint Plot":
                st.write("Generating Joint Plot (may take a moment)...")
                plt.close(fig)  # Close the default figure first
                joint_fig = sns.jointplot(data=data_to_plot, x=feature_x, y=feature_y, kind='scatter', alpha=0.5)
                joint_fig.fig.suptitle(f"Joint Plot: {feature_x} vs {feature_y}", y=1.02)
                st.pyplot(joint_fig.fig)
                plot_generated = True  # Special handling for jointplot

        elif x_is_num and not y_is_num:
            # Num (Y) vs Cat (X)
            # Limit categories shown
            order = data_to_plot[feature_y].value_counts().index[:15]
            rotation = 45 if data_to_plot[feature_y].nunique() > 5 else 0
            plot_type = st.selectbox("Choose Plot Type:", ["Box Plot", "Violin Plot", "Bar Plot (Mean)"])
            if plot_type == "Box Plot":
                sns.boxplot(data=data_to_plot, x=feature_y, y=feature_x, ax=ax, order=order)
                ax.set_title(f"Box Plot: {feature_x} by {feature_y}")
            elif plot_type == "Violin Plot":
                sns.violinplot(data=data_to_plot, x=feature_y, y=feature_x, ax=ax, order=order, inner='quartile')
                ax.set_title(f"Violin Plot: {feature_x} by {feature_y}")
            elif plot_type == "Bar Plot (Mean)":
                sns.barplot(data=data_to_plot, x=feature_y, y=feature_x, ax=ax, order=order, ci=None)  # ci=None faster
                ax.set_title(f"Bar Plot: Mean {feature_x} by {feature_y}")
            ax.tick_params(axis='x', rotation=rotation)


        elif not x_is_num and y_is_num:
            # Cat (X) vs Num (Y)
            # Limit categories shown
            order = data_to_plot[feature_x].value_counts().index[:15]
            rotation = 45 if data_to_plot[feature_x].nunique() > 5 else 0
            plot_type = st.selectbox("Choose Plot Type:", ["Box Plot", "Violin Plot", "Bar Plot (Mean)"])
            if plot_type == "Box Plot":
                sns.boxplot(data=data_to_plot, x=feature_x, y=feature_y, ax=ax, order=order)
                ax.set_title(f"Box Plot: {feature_y} by {feature_x}")
            elif plot_type == "Violin Plot":
                sns.violinplot(data=data_to_plot, x=feature_x, y=feature_y, ax=ax, order=order, inner='quartile')
                ax.set_title(f"Violin Plot: {feature_y} by {feature_x}")
            elif plot_type == "Bar Plot (Mean)":
                sns.barplot(data=data_to_plot, x=feature_x, y=feature_y, ax=ax, order=order, ci=None)  # ci=None faster
                ax.set_title(f"Bar Plot: Mean {feature_y} by {feature_x}")
            ax.tick_params(axis='x', rotation=rotation)


        else:
            # Cat vs Cat
            # Limit categories shown for hue
            hue_order = data_to_plot[feature_y].value_counts().index[:10]
            x_order = data_to_plot[feature_x].value_counts().index[:15]
            rotation = 45 if data_to_plot[feature_x].nunique() > 5 else 0
            plot_type = st.selectbox("Choose Plot Type:", ["Grouped Count Plot", "Stacked Bar Chart (Proportions)"])
            if plot_type == "Grouped Count Plot":
                sns.countplot(data=data_to_plot, x=feature_x, hue=feature_y, ax=ax, order=x_order, hue_order=hue_order)
                ax.set_title(f"Count Plot: {feature_x} grouped by {feature_y}")
            elif plot_type == "Stacked Bar Chart (Proportions)":
                # Calculate proportions
                crosstab = pd.crosstab(index=data_to_plot[feature_x], columns=data_to_plot[feature_y],
                                       normalize="index")
                crosstab = crosstab.loc[x_order, hue_order]  # Apply ordering/limits
                crosstab.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
                ax.set_title(f"Stacked Bar: Proportion of {feature_y} within {feature_x}")
                ax.legend(title=feature_y, bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.tick_params(axis='x', rotation=rotation)

        # Display the standard figure if not handled specially (like jointplot)
        if not plot_generated:
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as ex:
        st.error(f"Could not generate bivariate plot for {feature_x} vs {feature_y}: {ex}")
    finally:
        plt.close(fig)  # Close the plot to free memory
        gc.collect()


def multivariate_visualization(data, target, do_sample, n_sample_rows):
    """Handles multivariate feature visualization (Pairplot, Heatmap)."""
    st.subheader('🧬 Multivariate Analysis (Multiple Features)')

    data_to_plot = get_sampled_data(data, do_sample, n_sample_rows)
    num_cols = data_to_plot.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least two numerical columns for standard multivariate plots like Pairplot or Heatmap.")
        return

    chart_type = st.selectbox("Choose Multivariate Chart Type:", ["Pairplot", "Correlation Heatmap"])

    if chart_type == "Pairplot":
        st.write(f"Generating Pairplot for numerical features (using target '{target}' for coloring if suitable)...")
        st.info("Pairplots can be computationally intensive for many features or large datasets.")

        selected_num_cols = st.multiselect("Select numerical features for Pairplot:", num_cols,
                                           default=num_cols[:min(len(num_cols), 5)])  # Default to first 5

        if len(selected_num_cols) < 2:
            st.warning("Please select at least two numerical features for the Pairplot.")
            return

        hue_col = None
        if target and target in data_to_plot.columns:
            # Check if target is suitable for hue (categorical or low cardinality numeric)
            if is_object_dtype(data_to_plot[target]) or isinstance(data_to_plot[target].dtype, pd.CategoricalDtype):
                hue_col = target
            elif pd.api.types.is_numeric_dtype(data_to_plot[target]) and data_to_plot[target].nunique() < 15:
                # Arbitrary limit for numeric hue
                hue_col = target
            else:
                st.warning(
                    f"Target variable '{target}' has too many unique values or is not categorical; will not be used for"
                    f" coloring Pairplot.")

        if hue_col:
            cols_for_pairplot = selected_num_cols + [hue_col] if hue_col not in selected_num_cols else selected_num_cols
            st.write(f"Using '{hue_col}' for color (`hue`).")
            fig = sns.pairplot(data_to_plot[cols_for_pairplot], hue=hue_col, diag_kind='kde',
                               corner=True)  # Corner=True plots only lower triangle
        else:
            cols_for_pairplot = selected_num_cols
            fig = sns.pairplot(data_to_plot[cols_for_pairplot], diag_kind='kde', corner=True)

        fig.fig.suptitle("Pairplot of Selected Numerical Features", y=1.02)
        st.pyplot(fig)
        plt.close(fig.fig)  # Close plot
        gc.collect()

    elif chart_type == "Correlation Heatmap":
        correlation_heatmap_analysis(data_to_plot, target)  # Pass target for potential highlighting


def correlation_heatmap_analysis(data, target=None):
    """Calculates and displays a correlation heatmap for numerical features."""
    st.write("Generating Correlation Heatmap for numerical features...")
    num_data = data.select_dtypes(include=np.number)

    if num_data.shape[1] < 2:
        st.warning("Need at least two numerical features to calculate correlations.")
        return

    # Handle potential NaNs before correlation calculation
    try:
        imputer = SimpleImputer(strategy='median')  # Or 'mean'
        num_data_imputed = pd.DataFrame(imputer.fit_transform(num_data), columns=num_data.columns)
        corr_matrix = num_data_imputed.corr()
    except Exception as ex:
        st.error(f"Could not calculate correlation matrix, possibly due to NaN issues or column types: {ex}")
        return

    # Option to sort by correlation with target
    sort_by_target = False
    if target and target in corr_matrix.columns:
        # Checkbox needs a unique key if potentially reused in loop/rerun
        sort_by_target = st.checkbox(f"Sort heatmap by correlation with target ('{target}')?",
                                     key="heatmap_sort_checkbox")

    if sort_by_target:
        try:
            target_corr = corr_matrix[target].abs().sort_values(ascending=False)
            # Ensure index alignment after sorting
            corr_matrix = corr_matrix.loc[target_corr.index, target_corr.index]
            st.info(f"Sorted features by absolute correlation with '{target}'.")
        except KeyError:
            st.warning(f"Target '{target}' not found in numerical columns for sorting.")
        except Exception as ex:
            st.error(f"Error sorting heatmap: {ex}")

    fig, ax = plt.subplots(
        # This figsize calculation is functionally correct, though linters might warn
        figsize=(int(max(8.0, len(num_data.columns) * 0.6)), int(max(6.0, len(num_data.columns) * 0.5)))

    )
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                linewidths=.5, linecolor='lightgray', ax=ax, cbar=True, annot_kws={"size": 8})
    # Smaller font if many features
    plt.title("Correlation Matrix of Numerical Features")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Close plot
    gc.collect()


# -----------------------------
# Feature Importance Analysis
# -----------------------------
def feature_importance_analysis(data, target_column):
    st.subheader("Feature Importance Analysis")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    # For a categorical target, label-encode and use classification scores; for numerical, use regression scores.
    if y.dtype == 'object' or len(y.unique()) < 10:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        f_scores, _ = f_classif(X.select_dtypes(include=np.number), y_enc)
        mi_scores = mutual_info_classif(X.select_dtypes(include=np.number), y_enc)
    else:
        f_scores, _ = f_regression(X.select_dtypes(include=np.number), y)
        mi_scores = mutual_info_regression(X.select_dtypes(include=np.number), y)
    if X.select_dtypes(include=np.number).shape[1] > 0:
        pearson_scores = X.select_dtypes(include=np.number).corrwith(pd.Series(y)).abs()
    else:
        pearson_scores = pd.Series()
    importance_df = pd.DataFrame({
        'Feature': X.select_dtypes(include=np.number).columns,
        'F_score': f_scores,
        'Mutual_Info': mi_scores,
        'Pearson': pearson_scores
    })
    importance_df.sort_values(by='Mutual_Info', ascending=False, inplace=True)
    st.dataframe(importance_df)
    return importance_df


# -----------------------------
# Target Feature Selection
# -----------------------------
def target_feature(database):
    """Allows user to select a target feature."""
    st.write("Select the variable you want to predict or analyze feature importance against.")
    all_variables = ["None"] + list(database.columns)
    target: object = st.selectbox("Choose a Target Feature:", all_variables, index=0)
    return target if target != "None" else None


# function to find the type of problem that the dataset is having ( classification or regression)
def determine_problem_type(database, target_column):
    if target_column is None or target_column == "None":
        st.info("No target variable select. cannot determine problem type.")
        return None

    if target_column not in database.columns:
        st.error(f"Target column '{target_column}' not found in the processed data")
        return "Error: Target not found"

    target_series = database[target_column]
    #checking for the missing values in the target column which stop further processing
    if target_series.isnull().any():
        st.error(f"Target column '{target_column}' contains missing values AFTER cleaning. "
                 "Please review cleaning step. Cannot determine type reliably.")
        return "Error: Target has NaNs"

    # analyze data type and unique values
    col_dtype = target_series.dtype
    n_unique = target_series.nunique()
    n_rows = len(target_series)
    st.write(f"Analyzing Target: **`{target_column}`** (Type: {col_dtype}, Unique: {n_unique})")

    # constant target check
    if n_unique <= 1:
        st.warning(f"Target column '{target_column}' has <= 1 unique value. Cannot train supervised model.")
        return "Constant Target"

    # Boolean type
    if pd.api.types.is_bool_dtype(col_dtype):
        st.success(
            f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
            techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
            characteristics, the problem was identified as a **Binary Classification** (Boolean Target)** task.
            Accordingly, we proceeded with the appropriate supervised machine learning workflow tailored to 
            this problem type. """)

        return "Binary Classification"

    # object or categorical type:
    if pd.api.types.is_object_dtype(col_dtype) or isinstance(col_dtype, pd.CategoricalDtype):
        if n_unique == 2:

            st.success(
                f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
             techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
             characteristics, the problem was identified as a **Binary Classification** (Object/Categorical Target, 2 classes)** 
             task. Accordingly, we proceeded with the appropriate supervised machine learning 
             workflow tailored to this problem type. """)

            return "Binary Classification"
        else:
            if n_unique > 50 and n_rows > 0 and n_unique / n_rows > 0.5:
                st.warning(f"High unique values ({n_unique}) for object/category target. Is this an ID or text field?")
            st.success(
                f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
                techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
                characteristics, the problem was identified as a **Multiclass Classification** (Object/Categorical,
                {n_unique} classes)** task. Accordingly, we proceeded with the appropriate supervised machine learning 
                workflow tailored to this problem type. """)
            return "Multiclass Classification"

    # numeric type(int/ float)
    if pd.api.types.is_numeric_dtype(col_dtype):
        unique_values = target_series.unique()
        if n_unique == 2:
            is_binary_numeric = False
            if pd.api.types.is_float_dtype(col_dtype):
                val1, val2 = unique_values[0], unique_values[1]
                if (np.isclose(val1, 0) and np.isclose(val2, 1)) or \
                        (np.isclose(val1, 1) and np.isclose(val2, 0)):
                    is_binary_numeric = True
            elif pd.api.types.is_integer_dtype(col_dtype):
                if set(unique_values) == {0, 1}:
                    is_binary_numeric = True
            if is_binary_numeric:
                st.success(f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
                techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
                characteristics, the problem was identified as a **Binary Classification** (Numeric 0/1 Target)** task.
                Accordingly, we proceeded with the appropriate supervised machine learning workflow tailored
                 to this problem type. """)
                return "Binary Classification"
            else:
                st.success(f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
                techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
                characteristics, the problem was identified as a **Binary Classification (Numeric Target with 
                2 unique values: {unique_values})** task. Accordingly, we proceeded with the appropriate 
                supervised machine learning workflow tailored to this problem type. """)
                return "Binary Classification"

        # logic for float and int for regression and classification
        if pd.api.types.is_float_dtype(col_dtype):
            float_classification_threshold = 5
            if n_unique <= float_classification_threshold:
                st.warning(
                    f"Float target interpreted as **Multiclass Classification** (Very low unique values: {n_unique})")
                return "Multiclass Classification"
            else:
                st.success(f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
                techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
                characteristics, the problem was identified as a ****Regression** (Float target, high unique values)** 
                task. Accordingly, we proceeded with the appropriate supervised machine learning workflow 
                tailored to this problem type. """)
                return "Regression"

        elif pd.api.types.is_integer_dtype(col_dtype):
            # Thresholds for Classification vs Regression
            unique_value_threshold = 15
            unique_ratio_threshold = 0.05
            #  Ensure n_rows is not zero to avoid division error
            ratio = (n_unique / n_rows) if n_rows > 0 else 0
            is_likely_classification = (n_unique < unique_value_threshold) or \
                                       (ratio < unique_ratio_threshold)
            if is_likely_classification:
                st.warning(f"Numeric target interpreted as **Classification** (Low unique values: {n_unique})")
                return "Multiclass Classification"
            else:  # High unique integers -> Regression
                st.success(f"""Inferred Problem Type: We applied a series of automated data analysis and preprocessing 
                techniques to the given dataset with respect to the target variable **{target_column}**. Based on its
                 characteristics, the problem was identified as a **Regression (Numeric target, high unique values)** 
                 task. Accordingly, we proceeded with the appropriate supervised machine learning workflow 
                 tailored to this problem type. """)

                return "Regression"

        # fallback
        st.error(f"Could not determine problem type for target '{target_column}' (dtype {col_dtype}).")
        return "Error: Unknown Type"


#------------------------
# data splitting function
#------------------------

def data_split(data, target_column, test_sizes, stratifies, random_states):
    """
        Splits the data into training and testing sets.

        Args:
            data: The final processed DataFrame.
            target_column: The name of the target variable column.
            test_sizes: The proportion of the dataset to include in the test split.
            stratifies: Whether to stratify the split based on the target variable.
            random_states: Controls the shuffling applied to the data before splitting.

        Returns:
            A tuple containing (X_train, X_test, y_train, y_test), or None if an error occurs.
        """
    try:
        st.write("Splitting data...")
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # if stratification is possible
        problem_types = st.session_state.get("problem_type", None)
        stratify_option = None
        if stratifies and problem_types and "Classification" in problem_types:
            min_class_count = y.value_counts().min()
            if min_class_count < 2:
                st.warning(f"Cannot stratify : the smallest class in '{target_column}' has only {min_class_count} "
                           f"samples. Need at least 2. ")
            else:
                stratify_option = y
                st.write("Applying stratification based on Target variable.")
        elif stratifies and problem_types == "Regression":
            st.warning("Stratification is typically not used for regression tasks.")
        elif stratifies:
            st.warning("Problem type not determined as classification ")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_sizes,
            random_state=random_states,
            stratify=stratify_option
        )
        st.success("✅ Data split successfully!")
        return X_train, X_test, y_train, y_test

    except ValueError as vle:
        if "The least populated class" in str(vle):
            st.error(
                f"Stratification Error: {ve}. There are not enough samples in one of the classes ({target_column}) "
                f"to perform a stratified split with the chosen test size. Try disabling stratification or using a"
                f" different test size.")
        else:
            st.error(f"Error during data splitting: {vle}")
        return None

    except Exception as ex:
        st.error(f"An error occurred during data splitting: {ex}")
        return None

# ==============================================================================
#                            ML Model Definitions
# ==============================================================================


# define models suitable for classification
Classification_Models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB()
}

# define models suitable for regression
Regression_Models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor (SVR)": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
}


# ==============================================================================
#                      Model Training & Evaluation Function
# ==============================================================================
def train_evaluate_models(models_to_run, X_train, y_train, X_test, y_test, prob_type):
    """
        Trains selected models, evaluates them, and returns results.

        Args:
            models_to_run (dict): Dictionary of model instances {name: model_object}.
            X_train, y_train: Training data.
            X_test, y_test: Testing data.
            :param models_to_run:
            :param X_test:
            :param prob_type:
            :param prob_type:
            :param y_test:
            :param y_test:
            :param y_train:
            :param X_train:
            prob_type (str): "Binary Classification", "Multiclass Classification", or "Regression".

        Returns:
            tuple: (trained_models, evaluation_results)
                   trained_models (dict): {name: fitted_model_object}
                   evaluation_results (dict): {name: {metric_name: value}}

        """
    trained_models = {}
    evaluation_results = {}
    model_errors = {}
    if not models_to_run:
        st.warning("No model selected for training.")
        return {}, {}

    total_models = len(models_to_run)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models_to_run.items()):
        status_text.text(f"Training {name} ({i+1}/{total_models})..")
        try:
            # Training model
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            y_pred_proba_pos = None
            if hasattr(model, "predict_proba") and prob_type != "Regression":
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    # for binary classification to get probability of positive class(class 1)
                    if prob_type == "Binary Classification" and y_pred_proba.shape[1] == 2:
                        y_pred_proba_pos = y_pred_proba[:, 1]
                except Exception as exe:
                    st.warning(f"could not calculate prediction probabilities for {name}: {exe}")


            Metrics = {}
            if "Classification" in prob_type:
                Metrics["Accuracy"] = accuracy_score(y_test, y_pred)
                Metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                Metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                Metrics["F1-Score"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                # ROC AUC requires probability scores
                if prob_type == "Binary Classification" and y_pred_proba_pos is not None:
                    try:
                        Metrics["ROC AUC"] = roc_auc_score(y_test, y_pred_proba_pos)
                    except ValueError as roc_err:
                        Metrics["ROC AUC"] = f"N/A (str{roc_err})"  # Handle cases like only one class present
                elif prob_type == "Multiclass Classification" and y_pred_proba is not None:
                    try:
                        Metrics["ROC AUC (OvR)"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except ValueError as roc_err:
                        Metrics["ROC AUC (OvR"]= f"N/A (str{roc_err})"

                # storing confusion metrics data
                Metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
                try:
                    Metrics["Classification Report"] = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
                except ValueError as val_err:
                    Metrics["Classification Report"] = f"Could not generate report: {str(val_err)}"

            elif prob_type == "Regression":
                Metrics["R-squared (R²)"] = r2_score(y_test, y_pred)
                Metrics["Mean Absolute Error (MAE)"] = mean_absolute_error(y_test, y_pred)
                Metrics["Mean Squared Error (MSE)"] = mean_squared_error(y_test, y_pred)
                Metrics["Root Mean Squared Error (RMSE)"] = np.sqrt(mean_squared_error(y_test, y_pred))

            evaluation_results[name] = Metrics
            status_text.text(f"✅ Trained & Evaluated {name}")
        except Exception as ex:
            error_msg = f"Failed to train/evaluate {name}:{ex}"
            st.error(error_msg)
            model_errors[name] = str(ex)
        progress_bar.progress((i +1) / total_models)
    status_text.text("Model training and evaluation complete!")
    if model_errors:
        st.warning("Some models failed during the process.")
        with st.expander("Show Failed Model Errors"):
            st.json(model_errors)
    return trained_models, evaluation_results


# ==============================================================================
#                      Cross-Validation Function
# ==============================================================================
def perform_cross_validation(models_to_run, X_train, y_train, prob_type, CV_folds=5):
    """
    Performs cross-validation on the training data for selected models.

    Args:
        models_to_run (dict): Dictionary of model instances {name: model_object}.
        X_train, y_train: Training data.
        prob_type (str): Classification or Regression type.
        cv_folds (int): Number of cross-validation folds.
        :param CV_folds:
        :param prob_type:
        :param y_train:
        :param models_to_run:
        :param X_train:

    Returns:
        pd.DataFrame: DataFrame containing average CV scores for each model.

    """
    cv_results = {}
    model_errors_cv = {}

    if not models_to_run:
        st.warning("No models selected for Cross-Validation.")
        return pd.DataFrame()

    # Define CV splitter
    if "Classification" in prob_type:
        # Check if stratification is possible
        try:
            unique_classes = np.unique(y_train)
            min_class_count = y_train.value_counts().min()

            if len(unique_classes) < 2:
                st.warning("Cannot perform stratified CV: Only one class present in training data.")
                cv_splitter = KFold(n_splits=CV_folds, shuffle=True, random_state=42)
            elif min_class_count < CV_folds:
                st.warning(f"Smallest class count ({min_class_count}) is less than cv_folds ({CV_folds}). "
                           f"Using standard KFold instead of StratifiedKFold for CV.")
                cv_splitter = KFold(n_splits=CV_folds, shuffle=True, random_state=42)
            else:
                cv_splitter = StratifiedKFold(n_splits=CV_folds, shuffle=True, random_state=42)
        except Exception as ex:
            st.warning(f"Error setting up stratified CV: {ex}. Using standard KFold instead.")
            cv_splitter = KFold(n_splits=CV_folds, shuffle=True, random_state=42)

        # Define scoring metrics for classification
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0)
        }

        # Add ROC AUC for binary classification if applicable
        if prob_type == "Binary Classification" and len(np.unique(y_train)) == 2:
            scoring['roc_auc'] = 'roc_auc'

    elif prob_type == "Regression":
        cv_splitter = KFold(n_splits=CV_folds, shuffle=True, random_state=42)
        # Define scoring metrics for regression
        scoring = {
            'r2': make_scorer(r2_score),
            'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
            'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
        }
    else:  # Constant target or error
        st.error(f"Cannot perform Cross-Validation due to unsuitable problem type: {prob_type}")
        return pd.DataFrame()

    total_models = len(models_to_run)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models_to_run.items()):
        status_text.text(f"Running Cross-Validation for {name} ({i + 1}/{total_models})...")
        try:
            # Check if model supports the required functionality
            if hasattr(model, "fit") and hasattr(model, "predict"):
                # Perform cross-validation with timeout protection
                with st.spinner(f"CV fold calculations for {name}..."):
                    scores = cross_validate(
                        model, X_train, y_train,
                        cv=cv_splitter,
                        scoring=scoring,
                        n_jobs=-1,
                        error_score='raise',
                        return_train_score=True  # Also get training scores to check for overfitting
                    )

                # Store average scores - both test and train
                avg_test_scores = {f"CV_{key_s.replace('test_', '')}_mean": np.mean(scores[key_s])
                                   for key_s in scores if key_s.startswith('test_')}
                std_test_scores = {f"CV_{key_s.replace('test_', '')}_std": np.std(scores[key_s])
                                   for key_s in scores if key_s.startswith('test_')}

                avg_train_scores = {f"CV_train_{key_s.replace('train_', '')}_mean": np.mean(scores[key_s])
                                    for key_s in scores if key_s.startswith('train_')}

                # Handle negative scores (like neg_mae, neg_mse) - make them positive for easier interpretation
                for key_s in list(avg_test_scores.keys()) + list(avg_train_scores.keys()):
                    if 'neg_' in key_s:
                        if key_s in avg_test_scores:
                            avg_test_scores[key_s.replace('neg_', '')] = -avg_test_scores.pop(key_s)
                        elif key_s in avg_train_scores:
                            avg_train_scores[key_s.replace('neg_', '')] = -avg_train_scores.pop(key_s)

                for key_s in list(std_test_scores.keys()):
                    if 'neg_' in key_s:
                        std_test_scores[key_s.replace('neg_', '')] = std_test_scores.pop(key_s)

                # Add fit times
                fit_time_info = {
                    "CV_fit_time_mean": np.mean(scores['fit_time']),
                    "CV_fit_time_std": np.std(scores['fit_time']),
                    "CV_score_time_mean": np.mean(scores['score_time']),
                    "CV_score_time_std": np.std(scores['score_time'])
                }

                cv_results[name] = {**avg_test_scores, **std_test_scores, **avg_train_scores, **fit_time_info}
                status_text.text(f"✅ Cross-Validation complete for {name}")
            else:
                error_msg = f"{name} does not support required fit/predict methods"
                model_errors_cv[name] = error_msg
                st.warning(error_msg)
        except Exception as E:
            error_msg = f"Cross-Validation failed for {name}: {E}"
            st.error(error_msg)
            model_errors_cv[name] = str(E)

        progress_bar.progress((i + 1) / total_models)

    status_text.text("Cross-Validation runs complete!")
    if model_errors_cv:
        st.warning("Some models failed during Cross-Validation.")
        with st.expander("Show Failed CV Model Errors"):
            st.json(model_errors_cv)

    # Return empty DataFrame if no results
    if not cv_results:
        return pd.DataFrame()

    # Create a nice DataFrame with the results
    cv_df = pd.DataFrame(cv_results).T  # Transpose for models as rows

    # If we have both train and test scores, add overfitting indicator columns
    for col in cv_df.columns:
        if col.startswith("CV_") and not col.startswith("CV_train_") and not col.startswith(
                "CV_fit_time") and not col.startswith("CV_score_time"):
            train_col = f"CV_train_{col.split('CV_')[1]}"
            if train_col in cv_df.columns:
                # Calculate overfitting as difference between train and test scores
                cv_df[f"overfit_{col.split('CV_')[1]}"] = cv_df[train_col] - cv_df[col]

    return cv_df


# -----------------------------
# Main Execution Flow
# -----------------------------
upload_file = st.file_uploader(label="Choose a File..", type=['csv', 'xlsx', 'xls'], key="file_uploader")

# Use session state to store data across reruns
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'selected_target' not in st.session_state:
    st.session_state.selected_target = None
if "loaded_file_id" not in st.session_state:
    st.session_state.loaded_file_id = None
if 'encoding_method_index' not in st.session_state:
    st.session_state.encoding_method_index = 0
if 'scaling_option_index' not in st.session_state:
    st.session_state.scaling_option_index = 0
if 'manual_scaler_type_index' not in st.session_state:
    st.session_state.manual_scaler_type_index = 0

if upload_file is not None:
    if st.session_state.loaded_file_id != upload_file.file_id:
        st.info(f"New file detected: '{upload_file.name}'. Loading and resetting state...")
        try:
            loaded_data = load_dataset(upload_file)
            if loaded_data is not None:
                loaded_data.columns = loaded_data.columns.str.strip().str.replace(r'[^\w\s]+', '',
                                                                                  regex=True).str.replace(' ',
                                                                                                          '_').str.lower()
                st.session_state.dataset = loaded_data

                # reset all downstream state for the new file
                st.session_state.cleaned_data = None
                st.session_state.selected_target = None
                st.session_state.loaded_file_id = upload_file.file_id
                st.session_state.encoding_method_index = 0
                st.session_state.scaling_option_index = 0
                st.session_state.manual_scaler_type_index = 0
                st.success("State reset for new file")
            else:
                st.error("Failed to load the dataset from the uploaded file.")
                # Clear potentially inconsistent state if loading fails
                st.session_state.dataset = None
                st.session_state.loaded_file_id = None
                st.stop()
        except Exception as e:
            st.error(f"Error loading or processing the new file:{e}")
            st.session_state.dataset = None
            st.session_state.loaded_file_id = None
            st.stop()

    if st.session_state.dataset is not None:
        current_data = st.session_state.dataset.copy()  # Start with the original loaded data for this run

        # --- 1. Data Exploration ---
        st.header("1. 🕵️ Data Quality Assessment")
        if current_data is not None:
            assess_data_quality(current_data)
        else:
            st.warning("Dataset not available for quality assessment")

        # --- 2. Data Cleaning ---
        cleaned_data = None
        if current_data is not None:
            try:
                cleaned_data = handle_missing_values(current_data)
                st.session_state.cleaned_data = cleaned_data  # Store the result
            except Exception as e:
                st.error(f"Error during data cleaning: {e}")
                st.stop()  # Stopping is safer if cleaning is essential
        else:
            st.warning("No data available for cleaning.")

        # --- 3. Feature Engineering ---
        processed_data_local = st.session_state.get("cleaned_data")
        if processed_data_local is not None:
            try:
                st.write("Applying feature engineering..")
                endcoded_df = encoding_categorical_features(processed_data_local.copy())
                scaled_df = feature_scaling(endcoded_df)
                st.session_state.processed_data = scaled_df
                st.success("✅ Feature Engineering steps applied.")
            except Exception as e:
                st.error(f"An error occurred during Feature Engineering: {e}")
                st.session_state.processed_data = None
        else:
            st.warning("Skipping Feature engineering: Cleaning step did not product valid data.")
            st.session_state.processed_data = None

        # --- Target Selection ---
        st.header("🎯 Select Target Variable")

        if st.session_state.get('processed_data') is not None:
            # Allow selecting target based on the *final processed data*
            # Ensure target_feature function exists and takes the dataframe
            processed_data_for_target = st.session_state.processed_data
            selected_target_variable = target_feature(processed_data_for_target)

            #update session state and problem type if target changes
            if st.session_state.selected_target != selected_target_variable:
                st.session_state.selected_target = selected_target_variable
                st.session_state.problem_type = None

            targets = st.session_state.selected_target
            if targets:
                st.success(f"Target variable selected: `{targets}`")
            else:
                st.info("No target variable selected yet.")
        else:
            st.warning("cannot select target: Feature engineering must complete successfully first")
            st.session_state.selected_target = None

        # ------Determine problem type ----------
        st.header("🧠 Determine ML Problem Type")
        if st.session_state.get("processed_data") is not None and st.session_state.get('selected_target'):
            if st.session_state.get("problem_type") is None:
                st.write("Attempting to infer ML problem type...")
                problem_type_result = determine_problem_type(
                    st.session_state.processed_data,
                    st.session_state.selected_target
                )
                st.session_state.problem_type = problem_type_result
            if st.session_state.get("problem_type") and "Error" not in st.session_state.problem_type:
                st.success(f"✅ Problem type inferred as: **{st.session_state.problem_type}**")
            elif st.session_state.get('problem_type'):
                st.error(f"Problem type determination result:{st.session_state.problem_type}. Cannot proceed with standard modeling.")
        elif st.session_state.get('processed_data') is None:
            st.info("Waiting for Feature Engineering to complete...")
        else:
            st.info("Select a target variable to determine the problem type.")


        # --- 4. Data Visualization ---
        st.header("4. 📊 Data Visualization")
        st.write("Visualizations are performed on the **processed** (cleaned, encoded, scaled) data.")
        # Sampling options
        use_sampling = st.checkbox("Use Sampling for Visualizations (Recommended for large datasets)", value=False,
                                    key="viz_sampling_cb")
        sample_size = 1000  # Default sample size
        viz_data = st.session_state.processed_data
        if use_sampling:
            max_samples = min(50000, len(st.session_state.processed_data))
            default_sample = min(1000, max_samples)
            sample_size = st.number_input("Sample Size:", min_value=min(100, max_samples), max_value=max_samples,
                                              value=default_sample, step=100, key="viz_sample_size")
            if sample_size < len(st.session_state.processed_data):
                viz_data = st.session_state.processed_data.sample(n=sample_size, random_state=42)  # Use sampling
                st.write(f"Visualizing a sample of {sample_size} rows.")
            else:
                st.write("Sample size is equal to or larger than dataset size. Using full data.")

        viz_type = st.radio("Select the type of visualization:", ["Univariate", "Bivariate", "Multivariate"],
                                key="viz_type_radio")
        try:
            targets = st.session_state.selected_target
            if viz_type == "Univariate":
                univariate_visualization(st.session_state.processed_data, use_sampling, sample_size)
            elif viz_type == "Bivariate":
                if targets:
                    st.info(f"Selected Target for potential use: '{targets}'")
                    bivariate_visualization(st.session_state.processed_data, targets, use_sampling, sample_size)
                else:
                    st.warning("Please select a target variable for bivariate visualization")

            elif viz_type == "Multivariate":
                multivariate_visualization(st.session_state.processed_data, targets, use_sampling, sample_size)
        except Exception as e:
            st.error(f"Error during {viz_type} visualization: {e}")
        else:
            st.warning("Cannot perform Visualization: Feature Engineering must be completed first")


        # --- 5. Feature Importance ---
        if st.session_state.get('processed_data') is not None and st.session_state.get('selected_target'):
            # Call analysis function (Ensure it uses st.session_state.problem_type internally if needed)
            if st.session_state.get(
                    'problem_type') and "Error" not in st.session_state.problem_type and st.session_state.problem_type != "Constant Target":
                try:
                    feature_importance_analysis(st.session_state.processed_data, st.session_state.selected_target)
                except ValueError as ve:
                    st.error(f"Feature importance analysis failed : {ve}. Check for NaNs.")
                except Exception as e:
                    st.error(f"Error during Feature Importance Analysis: {e}")
            else:
                st.warning(
                    f"Cannot run Feature Importance. Problem Type: {st.session_state.get('problem_type', 'Not Determined')}")

        elif st.session_state.get('processed_data') is None:
            st.warning("Cannot perform Feature Importance: Feature Engineering must complete first.")
        else:
            st.info("Select a target variable to view Feature Importance.")

        # gc.collect()  # Garbage collect at the end of the run

        #new: ---optional feature selection and dropping part-----
        st.header("Feature Selection (optional)")
        if st.session_state.get("processed_data") is not None and st.session_state.get("selected_target"):
            st.write("Based on Feature importance and Domain knowledge, you can select a subset of features to use for modeling.")
            processed_data_local = st.session_state.processed_data
            target_col_local = st.session_state.selected_target
            processed_data_cols = processed_data_local.drop(columns=[target_col_local]).columns.tolist()

            # initializing if not present , default to all features
            if 'features_to_use' not in st.session_state:
                st.session_state.features_to_use = processed_data_cols
            current_selection = st.session_state.get("features_to_use", processed_data_cols)
            valid_selection = [col for col in current_selection if col in processed_data_cols]
            if not valid_selection:
                valid_selection = processed_data_cols
                st.session_state.features_to_use = valid_selection

            selected_features = st.multiselect(
                "Select features to use for model training: ",
                options=processed_data_cols,
                default=valid_selection,
                key="feature_selector_ms",
                help="Select the features you want to include in the training dataset from the feature importance part, "
                    "select those which have high score"
            )

            #updating session state only if selection is valid
            if selected_features:
                if set(selected_features) != set(st.session_state.features_to_use):
                    st.session_state.features_to_use = selected_features
                    # Clear downstream ML state if features change, as splits/models become invalid
                    keys_to_clear_on_feature_change = ['X_train', 'X_test', 'y_train', 'y_test', 'trained_models',
                                                           'evaluation_results', 'cv_results']
                    st.warning("Feature selection changed. Clearing previous data splits and model results")
                    for key in keys_to_clear_on_feature_change:
                        if key in st.session_state: del st.session_state[key]
                    st.rerun()
                st.info(f"{len(st.session_state.features_to_use)} features selected for model training")
            else:
                st.warning("No features selected! Please select at least one feature to proceed")
                if st.session_state.features_to_use:
                    st.session_state.features_to_use = []
                    keys_to_clear_on_feature_change = ['X_train', 'X_test', 'y_train', 'y_test', 'trained_models',
                                                       'evaluation_results', 'cv_results']
                    st.warning("Feature selection cleared. Clearing previous data splits and model results.")
                    for key in keys_to_clear_on_feature_change:
                        if key in st.session_state: del st.session_state[key]
                    st.rerun()
        elif st.session_state.get('processed_data') is None:
            st.warning("Cannot select features: feature engineering must complete first")
        else:
            st.warning("Cannot select features: Target variable must be selected first.")

        #---- 6 ML Modeling Setup  ------
        st.header(" 🤖 Machine Learning Modeling")
        can_proceed_to_ml = False
        # can_proceed_to_split = False
        if st.session_state.get('processed_data') is not None:
            problem_type = st.session_state.get('problem_type')
            if problem_type and "Error" not in problem_type and problem_type != "Constant Target":
                # can_proceed_to_split = True
                can_proceed_to_ml = True
            else:
                st.warning(f"Cannot proceed to ML Modeling. Check Problem Type determination step. Current Type: {problem_type}")
        else:
            st.warning("Cannot proceed to ML Modeling: Feature Engineering must be completed first.")

        if can_proceed_to_ml:
            current_problem_type = st.session_state.problem_type

            st.subheader("Splitting Data into Training and Testing Sets")
            st.info(f"Preparing data split for **{current_problem_type}** problem.")

            #--- splitting parameters----
            col_split1, col_split2 = st.columns(2)
            with col_split1:
                # Test size slider
                test_size = st.slider("Test Set Size (%): ", min_value=10, max_value=50, value=25, step=1,
                                       key="test_size_slider",
                                       help="Percentage of data to hold out for testing the final model."
                                      )
                test_proportion = test_size/ 100.0

            with col_split2:
                # Random state input
                random_state = st.number_input("Random State:", min_value=0, value=42, step=1, key="random_state_input",
                                               help="Seed for random number generator to ensure reproducible splits."
                                               )

            #stratification checkbox - default depends on problem type
            default_stratify = ("Classification" in current_problem_type)
            stratify = st.checkbox(
                "Stratify Split (Recommended for Classification)?",
                value=default_stratify,
                key="stratify_checkbox",
                help="Ensure train/test set have the same class proportions"
            )

            # split button--
            if st.button("🚀 Split Data", key="split_data_button"):
                data_to_split = st.session_state.processed_data
                target_col = st.session_state.selected_target
                features_to_use = st.session_state.get("features_to_use")

                if data_to_split is not None and target_col and features_to_use:
                    st.write(f"splitting data using {len(features_to_use)} selected features...")
                    features_to_use = [f for f in features_to_use if f!= target_col]
                    data_filtered = data_to_split[features_to_use + [target_col]]

                    with st.spinner("Performing train/test split..."):
                        split_results = data_split(
                            data=data_filtered,
                            target_column=target_col,
                            test_sizes=test_proportion,
                            stratifies=stratify,
                            random_states=random_state
                        )
                        if split_results:
                            st.session_state.X_train, st.session_state.X_test, \
                                st.session_state.y_train, st.session_state.y_test = split_results
                            st.write("----Split Results----")
                            info_cols = st.columns(2)
                            with info_cols[0]:
                                st.write("Features (X):")
                                st.dataframe(st.session_state.X_train.head(3))  # Show head
                                st.metric("Training Features Shape", str(st.session_state.X_train.shape))
                                st.metric("Testing Features Shape", str(st.session_state.X_test.shape))
                            with info_cols[1]:
                                st.write("Target (y):")
                                st.dataframe(st.session_state.y_train.head(3))  # Show head
                                st.metric("Training Target Shape", str(st.session_state.y_train.shape))
                                st.metric("Testing Target Shape", str(st.session_state.y_test.shape))
                        else:
                            # Clear any potentially old split data if splitting failed
                            keys_to_clear = ['X_train', 'X_test', 'y_train', 'y_test']
                            for key in keys_to_clear:
                                if key in st.session_state:
                                    del st.session_state[key]
                else:
                    st.error("Cannot split: Processed data or target column is missing.")

            #---- Display status if data is already split ----
            if 'X_train' in st.session_state and 'y_train' in st.session_state:
                st.success("✅ Data is split and ready for model training.")
                st.info(f"Training set: {st.session_state.X_train.shape[0]} samples. Testing set:{st.session_state.X_test.shape[0]} samples.")
            st.markdown("----")

            # --- Step 6b: Model Selection and Training ---
            # Only show if data has been successfully split
            if 'X_train' in st.session_state and 'y_train' in st.session_state:
                st.subheader("Select Models and Train")
                st.info(f"Select models appropriate for **{current_problem_type}**.")
                available_models = {}
                if "Classification" in current_problem_type:
                    available_models = Classification_Models
                elif "Regression" in current_problem_type:
                    available_models = Regression_Models

                if available_models:
                    selected_model_names = st.multiselect(
                        label="Select Models",
                        options=list(available_models.keys()),
                        default=list(available_models.keys())[:2],
                        key="model_multiselect"
                    )
                    model_to_train = {name: available_models[name] for name in selected_model_names}

                    # Cross validation Ui and button to perform
                    st.markdown('-----')
                    st.header("Optional: Cross-Validation")
                    st.write("Run cross-validation on the **training set** for a more robust performance estimate")
                    col_cv1, col_cv2 = st.columns([1, 3])
                    with col_cv1:
                        run_cv = st.checkbox("Run Cross validation?", value=False, key="run_cv_cb")
                        cv_folds = st.number_input("CV Folds", min_value=2, max_value=10, value=5, step=1,
                                                   disabled=not run_cv, key="cv_folds_input")

                    # showing button only if checkbox is ticked
                    if run_cv:
                        if st.button("📊 Run Cross-Validation", key="cv_button"):
                            if model_to_train:
                                with st.spinner(f"Running {cv_folds}- Fold Cross-validation.. This might take some time."):
                                    cv_result_df = perform_cross_validation(
                                        models_to_run=model_to_train,
                                        X_train=st.session_state.X_train,
                                        y_train=st.session_state.y_train,
                                        prob_type=current_problem_type,
                                        CV_folds=cv_folds
                                    )
                                    if not cv_result_df.empty:
                                        st.session_state.cv_results = cv_result_df
                                        st.success("cross validation complete")
                                    else:
                                        st.warning("Cross-validation did not product result(check logs/ errors).")
                            else:
                                st.warning("Select models before running cross-validation.")
                    # display cv results
                    if "cv_results" in st.session_state and not st.session_state.cv_results.empty:
                        st.markdown("----")
                        st.subheader(f"Cross Validation Results ({cv_folds}-Fold Average on Training Data)")
                        st.dataframe(st.session_state.cv_results.style.format('{:.4f}'))
                        st.caption("Metrics are averages across CV folds. '_std' indicates standard deviation.")
                        st.caption(
                            "'_mean' refers to the test fold score within CV. '_train_mean' is the average score "
                            "on the folds used for training within CV (useful for checking overfitting).")

                    # Model Training button
                    if st.button("💪 Train Selected Models", key="train_models_button"):
                        if model_to_train:
                            with st.spinner("Training Model and evaluating... This may take some time."):
                                #calling the training and evaluation function
                                trained_models_dict, eval_result_dict = train_evaluate_models(
                                    models_to_run=model_to_train,
                                    X_train=st.session_state.X_train,
                                    y_train=st.session_state.y_train,
                                    X_test=st.session_state.X_test,
                                    y_test=st.session_state.y_test,
                                    prob_type=current_problem_type
                                )
                                # storing the result in session state
                                st.session_state.trained_models = trained_models_dict
                                st.session_state.evaluation_results = eval_result_dict
                                st.success("🚀 Model Training & Evaluation Complete!")
                                st.rerun()
                        else:
                            st.warning("Please select at lease one model to train")
                else:
                    st.error("No model defined for the determined problem type.")
                st.markdown("----------")

                # --- Step 6c: Display Evaluation Results ---
                if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
                    st.subheader("📊 Model Evaluation Results")
                    # excluding complex task like confusion matrix
                    results_for_df = {}
                    for model_name, metrics in st.session_state.evaluation_results.items():
                        scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray, list, dict))}
                        results_for_df[model_name] = scalar_metrics

                    if results_for_df:
                        results_df = pd.DataFrame(results_for_df).T
                        numeric_cols = results_df.select_dtypes(include=np.number).columns
                        # formating numeric columns(e.g. accuracy , f1, MAE, R2)
                        st.dataframe(results_df.style.format({col: "{:.4f}" for col in numeric_cols}))
                    else:
                        st.info("No scalar evaluation metrics available to display in the table.")

                    # --- detailed plots for selected models----
                    st.subheader("Detailed Evaluation Graphs")
                    # user can select one model to see detailed plots and graphs
                    evaluated_models = list(st.session_state.evaluation_results.keys())
                    if evaluated_models:
                        selected_model_for_plot = st.selectbox(
                            "Select Model for Detailed Plots:",
                            options=evaluated_models,
                            key="plot_model_select"
                        )
                        if selected_model_for_plot and selected_model_for_plot in st.session_state.trained_models:
                            model_objective = st.session_state.trained_models[selected_model_for_plot]
                            model_metrics = st.session_state.evaluation_results[selected_model_for_plot]
                            if "Classification" in current_problem_type:
                                # using confusion matrix for the evaluation
                                if "Confusion Matrix" in model_metrics and isinstance(model_metrics["Confusion Matrix"], np.ndarray):
                                    try:
                                        st.write(f"**Confusion Matrix ({selected_model_for_plot}):**")
                                        fig_cm, ax_cm = plt.subplots()
                                        cm_display = ConfusionMatrixDisplay(
                                            confusion_matrix=model_metrics["Confusion Matrix"],
                                            display_labels=model_objective.classes_
                                        )
                                        cm_display.plot(ax=ax_cm, cmap="Blues", xticks_rotation='vertical')
                                        st.pyplot(fig_cm)
                                        plt.close(fig_cm)
                                    except Exception as e:
                                        st.error(f"could not plot confusion matrix: {e}")

                                # ROC curve
                                if current_problem_type == "Binary Classification" and "ROC AUC" in model_metrics and isinstance(model_metrics["ROC AUC"], float):
                                    try:
                                        st.write(f"**ROC Curve ({selected_model_for_plot}):**")
                                        fig_roc, ax_roc = plt.subplots()
                                        roc_display = RocCurveDisplay.from_estimator(
                                            model_objective, st.session_state.X_test, st.session_state.y_test, ax=ax_roc,
                                            name=selected_model_for_plot
                                        )
                                        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
                                        ax_roc.legend()
                                        st.pyplot(fig_roc)
                                        plt.close(fig_roc)
                                    except Exception as e:
                                        st.error(f"could not plot ROC Curve: {e}")
                                elif "ROC AUC (OvR)" in model_metrics and isinstance(model_metrics["ROC AUC (OvR)"], float):
                                    st.info(
                                        "ROC Curve plotting for multi-class is more complex "
                                        "(requires one-vs-rest plots per class). Showing AUC value in table.")

                                # Classification Report Text
                                if "Classification Report" in model_metrics:
                                    st.write(f"**Classification Report ({selected_model_for_plot}):**")
                                    st.text(model_metrics["Classification Report"])

                            elif current_problem_type == "Regression":
                                st.write(f"**Regression Evaluation ({selected_model_for_plot}):**")
                                # graph of model prediction vs actual
                                fig_reg, ax_reg = plt.subplots()
                                y_predict_reg = model_objective.predict(st.session_state.X_test)
                                sns.scatterplot(x=st.session_state.y_test, y=y_predict_reg, ax=ax_reg, alpha=0.6)
                                # using session state y_test to avoid any error
                                ax_reg.plot(
                                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                                    [st.session_state.y_test.min(), st.session_state.y_test.max()],
                                    'r--', lw=2
                                )
                                ax_reg.set_xlabel("Actual Values")
                                ax_reg.set_ylabel("Predicted Values")
                                ax_reg.set_title("Actual vs. Predicted Values")
                                st.pyplot(fig_reg)
                                plt.close(fig_reg)

                st.markdown('----')
                st.subheader("💾 Download Trained Model")
                if 'trained_models' in st.session_state and st.session_state.trained_models:
                    downloadable_models = list(st.session_state.trained_models.keys())
                    model_to_download = st.selectbox(
                        "Select model to download:",
                        options=downloadable_models,
                        key='model_download_select'
                    )
                    if model_to_download:
                        try:
                            model_object = st.session_state.trained_models[model_to_download]
                            buffer = io.BytesIO()
                            joblib.dump(model_object, buffer)
                            buffer.seek(0)

                            st.download_button(
                                f"Download {model_to_download} (.joblib)",
                                data=buffer,
                                file_name=f"{model_to_download.replace(' ', '_').lower()}_trained_model.joblib",
                                mime='application/octet-stream',  # generic binary file type
                                key=f"download_btn_{model_to_download}"
                            )
                            st.info("Click the button above to download the selected nodel file. ")
                        except Exception as e:
                            st.error(f"could not prepare model'{model_to_download}' for download: {e}")
                else:
                    st.info("No models have been trained in this session yet..")


            gc.collect()
        else:
            st.info("split the data first to evaluate model selection and training")
else:
    if st.session_state.loaded_file_id is not None:
        st.info("No file detected . Resetting application state")
        st.session_state.dataset = None
        st.session_state.cleaned_data = None
        st.session_state.selected_target = None
        st.session_state.loaded_file_id = None
        st.session_state.encoding_method_index = 0
        st.session_state.scaling_option_index = 0
        st.session_state.manual_scaler_type_index = 0
    else:
        st.info("Awaiting File Upload...")
