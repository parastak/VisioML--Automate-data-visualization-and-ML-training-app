"""
contains model definitions, ML logic( target selection , problem type id, split
train, evaluate , cv)
"""

import pandas as pd
# import numpy as np
import streamlit as st
# import gc
import joblib
import io

from typing import Dict, Tuple, Any, List
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report)

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB


# -----------------------------
# Target Feature Selection
# -----------------------------
def target_feature(database):
    """Allows user to select a target feature via Streamlit UI."""
    if database is None or database.empty:
        st.warning("Cannot select target: Data not available.")
        return None
    all_variables = ["None"] + list(database.columns)
    st.session_state.setdefault('target_column_index', 0)
    if st.session_state.target_column_index >= len(all_variables):
        st.session_state.target_column_index = 0

    selected_target_index = st.selectbox(
        "Choose a Target Feature:",
        options=range(len(all_variables)),
        format_func=lambda x: all_variables[x],
        index=st.session_state.target_column_index,
        key="target_select_unique_models"
    )
    st.session_state.target_column_index = selected_target_index
    selected_target = all_variables[selected_target_index]
    return selected_target if selected_target != "None" else None


#------------------------
# data splitting function
#------------------------
def data_split(data, target_column, test_sizes=0.2, stratifies=False, random_states=42, problem_type=None):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        test_sizes (float): Test set proportion (e.g., 0.2 = 20% test).
        stratifies (bool): Whether to apply stratification (classification tasks only).
        random_states (int): Random state for reproducibility.
        problem_type (str): Type of problem ("Classification", "Regression", etc.).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, status_message)
    """
    if target_column not in data.columns:
        return None, None, None, None, f"Error: Target column '{target_column}' not found in data."

    X = data.drop(columns=[target_column])
    y = data[target_column]
    stratify_option = None
    status_message = ""

    if stratifies and problem_type and "Classification" in problem_type:
        min_class_count = y.value_counts().min()
        if min_class_count < 2:
            status_message = f"Warning: Cannot stratify, class with minimum count has only {min_class_count} sample(s)."
        else:
            stratify_option = y
            status_message = "Info: Stratification applied."
    elif stratifies:
        status_message = "Note: Stratify ignored. Not a classification problem."

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_sizes,
            random_state=random_states,
            stratify=stratify_option
        )
        status_message += " Success: Data split completed."
        return X_train, X_test, y_train, y_test, status_message

    except ValueError as ve:
        if "least populated class" in str(ve):
            error_message = "Error: Stratified split failed due to insufficient samples in a class."
        else:
            error_message = f"Error during data splitting: {ve}"
        return None, None, None, None, error_message

    except Exception as ex:
        return None, None, None, None, f"Error: Unexpected error during splitting: {ex}"


# =======================================================
# ML Model Definitions & Training/Evaluation/CV Functions
# =======================================================

Classification_Models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB()
}

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


# noinspection PyTypedDict
def train_evaluate_models(models_to_run: Dict[str, Any], X_train, y_train, X_test, y_test, prob_type: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], str, List[str]]:
    """
    Trains and evaluates models.

    Args:
        models_to_run: Dict of model names and untrained estimators.
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        prob_type: "Classification", "Binary Classification", "Regression", etc.

    Returns:
        Tuple of:
            - trained_models: {model_name: trained_model}
            - evaluation_results: {model_name: {metrics}}
            - model_errors: {model_name: error}
            - final_message: summary message
            - status_updates: [log messages]
            :param prob_type:
            :param models_to_run:
            :param X_test:
            :param y_train:
            :param X_train:
            :param y_test:
    """
    trained_models = {}
    evaluation_results = {}
    model_errors = {}
    status_updates = []

    if not models_to_run:
        return {}, {}, {}, "No models selected for training.", []

    total_models = len(models_to_run)

    for i, (name, model) in enumerate(models_to_run.items()):
        status_updates.append(f"Training {name} ({i + 1}/{total_models})...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model

            if not hasattr(model, 'predict'):
                raise AttributeError(f"Model {name} does not have a predict method.")

            y_pred = model.predict(X_test)
            y_pred_proba = None
            y_pred_proba_pos = None

            if hasattr(model, "predict_proba") and "Regression" not in prob_type:
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    if prob_type == "Binary Classification" and y_pred_proba.shape[1] == 2:
                        y_pred_proba_pos = y_pred_proba[:, 1]
                except Exception as exe:
                    status_updates.append(f"Warning: Could not get probabilities for {name}: {type(exe).__name__} - {exe}")

            metrics = {}

            if "Classification" in prob_type:
                metrics["Accuracy"] = accuracy_score(y_test, y_pred)
                metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics["F1-Score"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                if prob_type == "Binary Classification" and y_pred_proba_pos is not None:
                    try:
                        metrics["ROC AUC"] = float(roc_auc_score(y_test, y_pred_proba_pos))
                    except ValueError as e:
                        metrics["ROC AUC"] = f"N/A ({e})"

                elif prob_type == "Multiclass Classification" and y_pred_proba is not None:
                    try:
                        metrics["ROC AUC (OvR)"] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'))
                    except ValueError as e:
                        metrics["ROC AUC (OvR)"] = f"N/A ({e})"

                metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred).tolist()  # serialize matrix
                try:
                    metrics["Classification Report"] = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
                except ValueError as err:
                    metrics["Classification Report"] = f"Could not generate report: {err}"

            elif "Regression" in prob_type:
                metrics["R²"] = r2_score(y_test, y_pred)
                metrics["MAE"] = mean_absolute_error(y_test, y_pred)
                metrics["MSE"] = mean_squared_error(y_test, y_pred)
                metrics["RMSE"] = np.sqrt(metrics["MSE"])

            evaluation_results[name] = metrics
            status_updates.append(f"✅ {name} trained and evaluated.")

        except Exception as ex:
            error_msg = f"❌ Failed {name}: {type(ex).__name__} - {ex}"
            model_errors[name] = str(ex)
            status_updates.append(error_msg)

    final_message = f"Training complete for {len(trained_models)} model(s)."
    if model_errors:
        final_message += f" {len(model_errors)} model(s) failed."

    return trained_models, evaluation_results, model_errors, final_message, status_updates


def perform_cross_validation(models_to_run, X_train, y_train, prob_type, cv_folds=5):
    """
    Perform cross-validation for classification or regression models.

    Args:
        models_to_run (dict): Dictionary of {model_name: model_instance}.
        X_train (DataFrame): Feature data.
        y_train (Series): Target data.
        prob_type (str): "Binary Classification", "Multiclass Classification", or "Regression".
        cv_folds (int): Number of CV folds.

    Returns:
        tuple: (cv_results_df, model_errors, summary_msg, status_log)
    """
    cv_results = {}
    model_errors_cv = {}
    status_updates = []

    if not models_to_run:
        return pd.DataFrame(), {}, "No models selected for Cross-Validation.", []

    # --- Splitter & Scoring Setup ---
    scoring = {}
    try:
        if "Classification" in prob_type:
            min_class_count = y_train.value_counts().min()
            if min_class_count < cv_folds:
                cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                status_updates.append("Warning: Using KFold (class imbalance or small classes).")
            else:
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                status_updates.append("Using StratifiedKFold.")
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
                'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
                'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0)
            }
            if prob_type == "Binary Classification" and len(np.unique(y_train)) == 2:
                scoring['roc_auc'] = 'roc_auc'
        elif prob_type == "Regression":
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = {
                'r2': make_scorer(r2_score),
                'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
                'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False)
            }
            status_updates.append("Using KFold for regression.")
        else:
            return pd.DataFrame(), {}, f"Error: Unsupported problem type: {prob_type}", []
    except Exception as ex:
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        status_updates.append(f"Warning: Fallback to KFold due to error: {ex}")

    # --- Cross-Validation ---
    for i, (name, model) in enumerate(models_to_run.items()):
        status_updates.append(f"Running CV for {name} ({i+1}/{len(models_to_run)})")
        try:
            if hasattr(model, "fit") and hasattr(model, "predict"):
                scores = cross_validate(
                    model, X_train, y_train,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score='raise',
                    return_train_score=True
                )

                # --- Extract scores ---
                avg_test_scores = {f"CV_{k.replace('test_', '')}_mean": np.mean(scores[k]) for k in scores if k.startswith('test_')}
                std_test_scores = {f"CV_{k.replace('test_', '')}_std": np.std(scores[k]) for k in scores if k.startswith('test_')}
                avg_train_scores = {f"CV_train_{k.replace('train_', '')}_mean": np.mean(scores[k]) for k in scores if k.startswith('train_')}

                # Make negative scores positive for interpretability
                for key in list(avg_test_scores.keys()):
                    if 'neg_' in key:
                        avg_test_scores[key.replace('neg_', '')] = -avg_test_scores.pop(key)
                for key in list(avg_train_scores.keys()):
                    if 'neg_' in key:
                        avg_train_scores[key.replace('neg_', '')] = -avg_train_scores.pop(key)
                for key in list(std_test_scores.keys()):
                    if 'neg_' in key:
                        std_test_scores[key.replace('neg_', '')] = std_test_scores.pop(key)

                fit_time_info = {
                    "CV_fit_time_mean": np.mean(scores['fit_time']),
                    "CV_fit_time_std": np.std(scores['fit_time']),
                    "CV_score_time_mean": np.mean(scores['score_time']),
                    "CV_score_time_std": np.std(scores['score_time']),
                }

                all_scores = {**avg_test_scores, **std_test_scores, **avg_train_scores, **fit_time_info}
                cv_results[name] = all_scores
                status_updates.append(f"✅ Finished CV for {name}")
            else:
                raise AttributeError(f"{name} lacks fit/predict methods.")
        except Exception as e:
            model_errors_cv[name] = str(e)
            status_updates.append(f"❌ CV failed for {name}: {e}")

    # --- Format Output ---
    if not cv_results:
        return pd.DataFrame(), model_errors_cv, "Cross-Validation failed for all models.", status_updates

    cv_df = pd.DataFrame(cv_results).T

    # Add overfitting detection
    for col in cv_df.columns:
        if col.startswith("CV_") and not col.startswith("CV_train_") and "time" not in col:
            train_col = f"CV_train_{col.split('CV_')[1]}"
            if train_col in cv_df.columns:
                cv_df[f"overfit_{col.split('CV_')[1]}"] = cv_df[train_col] - cv_df[col]

    summary_msg = "Cross-Validation complete."
    if model_errors_cv:
        summary_msg += f" {len(model_errors_cv)} model(s) had errors."

    return cv_df, model_errors_cv, summary_msg, status_updates


# ==============================================================================
# Model Download Helper (Added)
# ==============================================================================
def get_model_download_link(model_object, model_name):
    """Generates components for a Streamlit download button for a model."""
    try:
        buffer = io.BytesIO()
        joblib.dump(model_object, buffer)
        buffer.seek(0)
        file_name = f"{model_name.replace(' ', '_').lower()}_trained_model.joblib"
        mime="application/octet-stream"
        return buffer, file_name, mime
    except Exception as e:
        st.error(f"Could not prepare model '{model_name}' for download: {e}")
        return None, None, None
