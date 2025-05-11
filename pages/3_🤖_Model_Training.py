# pages/3_🤖_ML_Modeling.py

import streamlit as st
import pandas as pd
import numpy as np  # For metrics display or any array operations
import gc
import seaborn as sns
# import joblib  # For model download if not handled in util
# import io  # For model download if not handled in util

# Import from your utility modules
from utils.models import (target_feature, data_split, Classification_Models, Regression_Models,
                          train_evaluate_models, perform_cross_validation, get_model_download_link)
from utils.eda import determine_problem_type, feature_importance_analysis  # Assuming logic-only versions

# For plotting detailed evaluation results
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="ML Modeling & Evaluation")
st.title("🤖 Step 4: ML Modeling, Training & Evaluation")
st.write("Define your target, select features, split data, train models, evaluate, and download.")
st.divider()

# --- Prerequisite Checks ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.warning("⬅️ Please complete Data Cleaning and Feature Engineering first (check previous steps).")
    st.stop()

processed_df = st.session_state.processed_data

# --- 1. Target Variable Selection ---
with st.container():
    st.header("🎯 1. Select Target Variable for Modeling")
    # This target_feature function from your models.py already has the st.selectbox
    selected_target_on_page = target_feature(processed_df)

    # Update session state if target selection changes from any previous state
    if st.session_state.get('selected_target') != selected_target_on_page:
        st.session_state.selected_target = selected_target_on_page
        # CRITICAL: Reset all downstream states if target changes
        keys_to_reset = ['problem_type', 'features_to_use', 'X_train', 'X_test',
                         'y_train', 'y_test', 'trained_models',
                         'evaluation_results', 'cv_results', 'feature_selector_ms_default_modeling']
        for key in keys_to_reset:
            st.session_state.pop(key, None)  # Remove or set to None
        st.rerun()  # Use rerun to ensure UI updates correctly

    if not st.session_state.selected_target:
        st.warning("Please select a target variable to proceed with modeling.")
        st.stop()
    else:
        st.success(f"Target Variable for Modeling: **`{st.session_state.selected_target}`**")
st.divider()

# --- 2. Determine ML Problem Type ---
with st.container():
    st.header("🧠 2. Determine ML Problem Type")
    # Determine if not already set or if target changed
    if st.session_state.get('problem_type') is None and st.session_state.selected_target:
        problem_type_str, problem_msg_str = determine_problem_type(
            st.session_state.processed_data,
            st.session_state.selected_target
        )
        st.session_state.problem_type = problem_type_str
        st.info(problem_msg_str)  # Display the message from the function
    elif st.session_state.get('problem_type'):
        st.success(f"Inferred Problem Type: **{st.session_state.problem_type}**")

    current_problem_type_val = st.session_state.get('problem_type')
    if not current_problem_type_val or "Error" in current_problem_type_val or current_problem_type_val == "Constant Target":
        st.error(
            f"Cannot proceed with modeling. Problem Type: `{current_problem_type_val}`. Please check your target variable.")
        st.stop()
st.divider()

# --- 3. Feature Importance (Optional Display) ---
with st.expander("⭐ Show Feature Importance Scores (Guiding Selection)", expanded=False):
    if current_problem_type_val and st.session_state.selected_target:
        st.write(
            "Scores below are calculated on all numeric features relative to the target. Use this to guide your feature selection.")
        fi_df_result, fi_errors_result = feature_importance_analysis(  # From utils.eda
            processed_df,
            st.session_state.selected_target,
            current_problem_type_val
        )
        if fi_df_result is not None:
            st.dataframe(fi_df_result)
        if fi_errors_result:
            st.warning(f"Could not calculate all FI scores: {fi_errors_result}")
    else:
        st.caption("Select a target and determine problem type to see feature importance.")
st.divider()

# --- 4. Feature Selection ---
with st.container():
    st.header("🔬 4. Select Features for Modeling")
    # Exclude target from available features for selection
    all_possible_features_for_x = processed_df.drop(columns=[st.session_state.selected_target],
                                                    errors='ignore').columns.tolist()

    # Initialize or validate 'features_to_use' based on current possible features
    # 'feature_selector_ms_default_modeling' stores the actual list for the widget
    if 'feature_selector_ms_default_modeling' not in st.session_state or \
            not set(st.session_state.feature_selector_ms_default_modeling).issubset(set(all_possible_features_for_x)):
        st.session_state.feature_selector_ms_default_modeling = all_possible_features_for_x

    selected_features_from_ui = st.multiselect(
        "Select features TO USE for modeling (defaults to all):",
        options=all_possible_features_for_x,
        default=st.session_state.feature_selector_ms_default_modeling,
        key="modeling_page_feature_selector_ms"  # Unique key
    )

    # Update session_state.features_to_use if selection changes
    # and clear downstream results
    if set(selected_features_from_ui) != set(st.session_state.get('features_to_use', [])):
        st.session_state.features_to_use = selected_features_from_ui
        st.session_state.feature_selector_ms_default_modeling = selected_features_from_ui  # Update default for multiselect
        # Clear downstream states
        keys_to_reset_fs = ['X_train', 'X_test', 'y_train', 'y_test', 'trained_models', 'evaluation_results',
                            'cv_results']
        st.warning("Feature selection changed. Clearing previous data splits and model results.")
        for key_fs in keys_to_reset_fs:
            st.session_state.pop(key_fs, None)
        st.rerun()

    if not st.session_state.get('features_to_use'):
        st.error("No features selected. Please select at least one feature to proceed.")
        st.stop()
    else:
        st.info(f"**{len(st.session_state.features_to_use)} features** are currently selected for modeling.")
st.divider()

# --- 5. Data Splitting ---
with st.container():
    st.header("🔪 5. Split Data into Training and Test Sets")
    col_split1, col_split2 = st.columns(2)
    with col_split1:
        test_size_ui = st.slider("Test Set Size (%):", 10, 50, 25, 1, key="modeling_page_test_size_slider")
    test_proportion_ui = test_size_ui / 100.0
    with col_split2:
        random_state_ui = st.number_input("Random State for Splitting:", 0, 1000, 42, 1,
                                          key="modeling_page_random_state_input")

    default_stratify_ui = ("Classification" in current_problem_type_val)
    stratify_ui = st.checkbox("Stratify Split (recommended for classification)?", value=default_stratify_ui,
                              key="modeling_page_stratify_cb")

    if st.button("🚀 Split Data", key="modeling_page_split_data_btn"):
        data_for_split_input = processed_df
        target_column_name_input = st.session_state.selected_target
        features_selected_input = st.session_state.features_to_use

        try:
            # Ensure target is not accidentally in features_selected_input (it shouldn't be)
            features_for_X_input = [f for f in features_selected_input if f != target_column_name_input]
            # Prepare the DataFrame with only selected X features and the Y target
            data_filtered_for_splitting = data_for_split_input[features_for_X_input + [target_column_name_input]]

            X_tr_res, X_te_res, y_tr_res, y_te_res, split_msg_res = data_split(  # From utils.models
                data_filtered_for_splitting, target_column_name_input, test_proportion_ui, stratify_ui, random_state_ui
            )
            st.info(split_msg_res)  # Display status message from the function
            if X_tr_res is not None:
                st.session_state.X_train, st.session_state.X_test = X_tr_res, X_te_res
                st.session_state.y_train, st.session_state.y_test = y_tr_res, y_te_res
                st.success("Data split complete!")
                # Clear previous model results if data is re-split
                st.session_state.trained_models = None
                st.session_state.evaluation_results = None
                st.session_state.cv_results = None
                st.rerun()  # Rerun to show split status and enable next steps
            else:  # Error during split
                keys_to_clear_split_error = ['X_train', 'X_test', 'y_train', 'y_test']
                for key_split_err in keys_to_clear_split_error:
                    st.session_state.pop(key_split_err, None)
        except Exception as e_split:
            st.error(f"Error preparing data for splitting: {e_split}")

    if 'X_train' in st.session_state and st.session_state.X_train is not None:
        st.success("✅ Data is split and ready for model training.")
        col_shape1, col_shape2 = st.columns(2)
        with col_shape1:
            st.metric("Training Data Shape (X_train)", f"{st.session_state.X_train.shape}")
            st.metric("Training Target Shape (y_train)", f"{st.session_state.y_train.shape}")
        with col_shape2:
            st.metric("Test Data Shape (X_test)", f"{st.session_state.X_test.shape}")
            st.metric("Test Target Shape (y_test)", f"{st.session_state.y_test.shape}")
    else:
        st.info("Configure and click 'Split Data' to proceed.")
st.divider()

# --- 6. Model Training & Evaluation (Only if data is split) ---
if 'X_train' in st.session_state and st.session_state.X_train is not None:
    st.header("🛠️ 6. Model Training, Cross-Validation & Evaluation")

    # --- Model Selection ---
    st.subheader("Select Models to Train")
    available_models_for_problem = Classification_Models if "Classification" in current_problem_type_val else Regression_Models

    # Persist model selection
    st.session_state.setdefault('ml_page_selected_models_names',
                                list(available_models_for_problem.keys())[:1])  # Default to first model

    selected_model_names_from_ui = st.multiselect(
        "Choose models to train and evaluate:",
        options=list(available_models_for_problem.keys()),
        default=st.session_state.ml_page_selected_models_names,
        key="modeling_page_model_multiselect"
    )
    # If selection changes, update state (no need to clear downstream here, buttons will trigger re-runs)
    if set(selected_model_names_from_ui) != set(st.session_state.ml_page_selected_models_names):
        st.session_state.ml_page_selected_models_names = selected_model_names_from_ui
        # Clearing results if model selection changes might be too aggressive,
        # better to let CV/Train buttons re-trigger with new selection.

    models_to_run_for_training = {name: available_models_for_problem[name] for name in
                                  st.session_state.ml_page_selected_models_names}

    if not models_to_run_for_training:
        st.warning("Please select at least one model to train.")
    else:
        st.info(f"Selected models: {', '.join(models_to_run_for_training.keys())}")

    # --- Cross-Validation (Optional) ---
    with st.expander("📊 Perform Cross-Validation (on Training Set)", expanded=False):
        run_cv_ui_option = st.checkbox("Enable Cross-Validation?",
                                       value=st.session_state.get('modeling_page_run_cv', False),
                                       key="modeling_page_run_cv")
        cv_folds_ui_option = 5
        if run_cv_ui_option:
            cv_folds_ui_option = st.number_input("Number of CV Folds:", 2, 10, 5, 1, key="modeling_page_cv_folds_input")
            if st.button("📈 Run Cross-Validation", key="modeling_page_cv_btn"):
                if models_to_run_for_training:
                    with st.spinner(f"Running {cv_folds_ui_option}-Fold CV on training data... This may take a while."):
                        # Assuming perform_cross_validation is logic-only
                        cv_df_result, cv_errors_result, cv_msg_result, _ = perform_cross_validation(
                            models_to_run_for_training, st.session_state.X_train, st.session_state.y_train,
                            current_problem_type_val, cv_folds_ui_option
                        )
                        st.session_state.cv_results = cv_df_result  # Store CV results DataFrame
                        st.info(cv_msg_result)
                        if cv_errors_result:
                            st.warning("Cross-Validation encountered errors for some models:")
                            st.json(cv_errors_result)
                        if cv_df_result is not None and not cv_df_result.empty:
                            st.success("Cross-Validation complete!")
                        else:
                            st.warning("Cross-Validation did not produce results.")
                else:
                    st.warning("No models selected for Cross-Validation.")

        # Display CV results if they exist
        if 'cv_results' in st.session_state and st.session_state.cv_results is not None and not st.session_state.cv_results.empty:
            st.write("Cross-Validation Mean Scores (on Training Folds):")
            st.dataframe(st.session_state.cv_results.style.format("{:.4f}"))
            # You could add logic here to highlight best models or overfitting.
        elif run_cv_ui_option:
            st.caption("Click 'Run CV' to see results.")

    # --- Final Model Training and Evaluation on Test Set ---
    st.subheader("🚀 Train Final Models & Evaluate on Test Set")
    use_train_sample_ui_option = st.checkbox("Use a sample of training data for faster final training?",
                                             key="modeling_page_train_sample_cb")
    train_sample_size_ui_option = 10000  # Default
    if use_train_sample_ui_option:
        train_sample_size_ui_option = st.number_input(
            "Training sample size:", 100, len(st.session_state.X_train),
            min(10000, len(st.session_state.X_train)), key="modeling_page_train_sample_input"
        )

    if st.button("💪 Train & Evaluate Selected Models on Full Test Set", key="modeling_page_train_eval_btn"):
        if models_to_run_for_training:
            X_train_for_run, y_train_for_run = st.session_state.X_train, st.session_state.y_train
            if use_train_sample_ui_option and train_sample_size_ui_option < len(X_train_for_run):
                sample_indices = X_train_for_run.sample(n=train_sample_size_ui_option,
                                                        random_state=random_state_ui).index
                X_train_for_run = X_train_for_run.loc[sample_indices]
                y_train_for_run = y_train_for_run.loc[sample_indices]
                st.info(f"Training final models on a sample of {train_sample_size_ui_option} instances.")

            with st.spinner("Training final models and evaluating... This may take some time."):
                # Assuming train_evaluate_models is logic-only
                trained_models_res, eval_results_res, errors_train_eval, final_msg_train_eval, _ = train_evaluate_models(
                    models_to_run_for_training, X_train_for_run, y_train_for_run,
                    st.session_state.X_test, st.session_state.y_test, current_problem_type_val
                )
                st.session_state.trained_models = trained_models_res
                st.session_state.evaluation_results = eval_results_res
                st.info(final_msg_train_eval)
                if errors_train_eval:
                    st.warning("Training/Evaluation encountered errors for some models:")
                    st.json(errors_train_eval)
            st.success("Final Training & Evaluation on Test Set Complete!")
            st.rerun()  # Rerun to display results below cleanly
        else:
            st.warning("No models selected to train and evaluate.")
    st.divider()

    # --- Display Test Set Evaluation Results & Download Model ---
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        st.header("📊 Test Set Evaluation Results")

        results_for_table = {}
        for model_name, metrics in st.session_state.evaluation_results.items():
            # Extract only scalar metrics for the summary table
            scalar_metrics_for_table = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
            results_for_table[model_name] = scalar_metrics_for_table

        if results_for_table:
            results_df_display_final = pd.DataFrame(results_for_table).T
            # Identify numeric columns to format
            numeric_cols = results_df_display_final.select_dtypes(include=[np.number]).columns.tolist()

            # Apply formatting only to numeric columns
            st.dataframe(results_df_display_final.style.format({col: "{:.4f}" for col in numeric_cols}))

        else:
            st.info("No scalar metrics available to display in summary table from evaluation results.")

        # Detailed Plots for each model
        st.subheader("📈 Detailed Evaluation Plots per Model")
        evaluated_model_names_list = list(st.session_state.evaluation_results.keys())
        if evaluated_model_names_list:
            # Persist selection for plot model
            st.session_state.setdefault('modeling_page_plot_model_select_val', evaluated_model_names_list[0])
            if st.session_state.modeling_page_plot_model_select_val not in evaluated_model_names_list:  # Reset if invalid
                st.session_state.modeling_page_plot_model_select_val = evaluated_model_names_list[0]

            selected_model_for_detailed_plot = st.selectbox(
                "Select Model for Detailed Plots:",
                options=evaluated_model_names_list,
                index=evaluated_model_names_list.index(st.session_state.modeling_page_plot_model_select_val),
                key="modeling_page_plot_model_select_val_widget"  # Use the state key for the widget too
            )
            st.session_state.modeling_page_plot_model_select_val = selected_model_for_detailed_plot  # Update state

            if selected_model_for_detailed_plot and selected_model_for_detailed_plot in st.session_state.trained_models:
                model_object_for_plot = st.session_state.trained_models[selected_model_for_detailed_plot]
                model_metrics_for_plot = st.session_state.evaluation_results[selected_model_for_detailed_plot]

                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    if "Classification" in current_problem_type_val:
                        if "Confusion Matrix" in model_metrics_for_plot and isinstance(
                                model_metrics_for_plot["Confusion Matrix"], np.ndarray):
                            st.write(f"**Confusion Matrix for {selected_model_for_detailed_plot}**")
                            fig_cm, ax_cm = plt.subplots()
                            cm_display = ConfusionMatrixDisplay(
                                model_metrics_for_plot["Confusion Matrix"],
                                display_labels=model_object_for_plot.classes_  # Ensure classes_ is available
                            )
                            cm_display.plot(ax=ax_cm, cmap='Blues', xticks_rotation='vertical')
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)
                    elif current_problem_type_val == "Regression":
                        st.write(f"**Actual vs. Predicted for {selected_model_for_detailed_plot}**")
                        fig_reg, ax_reg = plt.subplots()
                        y_pred_reg_plot = model_object_for_plot.predict(st.session_state.X_test)
                        sns.scatterplot(x=st.session_state.y_test.squeeze(), y=y_pred_reg_plot, ax=ax_reg, alpha=0.6)
                        ax_reg.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                                    [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--', lw=2)
                        ax_reg.set_xlabel("Actual Values")
                        ax_reg.set_ylabel("Predicted Values")
                        st.pyplot(fig_reg)
                        plt.close(fig_reg)

                with plot_col2:
                    if current_problem_type_val == "Binary Classification":
                        if "ROC AUC" in model_metrics_for_plot and isinstance(model_metrics_for_plot["ROC AUC"],
                                                                              float):  # Check if metric exists
                            st.write(f"**ROC Curve for {selected_model_for_detailed_plot}**")
                            fig_roc, ax_roc = plt.subplots()
                            try:
                                RocCurveDisplay.from_estimator(model_object_for_plot, st.session_state.X_test,
                                                               st.session_state.y_test, ax=ax_roc,
                                                               name=selected_model_for_detailed_plot)
                                ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)  # Reference line
                                st.pyplot(fig_roc)
                            except Exception as e_roc:
                                st.warning(f"Could not plot ROC curve: {e_roc}")
                            finally:
                                plt.close(fig_roc)

                if "Classification" in current_problem_type_val and "Classification Report" in model_metrics_for_plot:
                    st.write(f"**Classification Report for {selected_model_for_detailed_plot}**")
                    st.text_area("", model_metrics_for_plot["Classification Report"], height=300,
                                 key=f"class_report_{selected_model_for_detailed_plot}")
        st.divider()

        # --- Model Download Section ---
        st.header("💾 Download Trained Model")
        downloadable_model_names = list(st.session_state.get('trained_models', {}).keys())
        if downloadable_model_names:
            st.session_state.setdefault('modeling_page_model_to_download_name', downloadable_model_names[0])
            if st.session_state.modeling_page_model_to_download_name not in downloadable_model_names:  # Reset if invalid
                st.session_state.modeling_page_model_to_download_name = downloadable_model_names[0]

            model_to_download_ui_name = st.selectbox(
                "Select model to download:",
                options=downloadable_model_names,
                index=downloadable_model_names.index(st.session_state.modeling_page_model_to_download_name),
                key="modeling_page_model_to_download_name_widget"
            )
            st.session_state.modeling_page_model_to_download_name = model_to_download_ui_name  # Update state

            if model_to_download_ui_name:
                model_object_to_download = st.session_state.trained_models[model_to_download_ui_name]
                # Use the get_model_download_link from utils.models
                model_buffer, model_file_name, model_mime = get_model_download_link(model_object_to_download,
                                                                                    model_to_download_ui_name)
                if model_buffer:
                    st.download_button(
                        label=f"Download {model_to_download_ui_name} (.joblib)",
                        data=model_buffer,
                        file_name=model_file_name,
                        mime=model_mime,
                        key=f"download_btn_{model_to_download_ui_name.replace(' ', '_')}"
                    )
        else:
            st.info("No models have been trained in this session yet to download.")

gc.collect()  # Final garbage collection for the page
