import pandas as pd
import numpy as np
import streamlit as st
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pandas.api.types import is_object_dtype


# ============================================
# Feature Visualizing Functions
# ============================================


def get_sampled_data(data, do_sample, n_sample_row):
    """
    samples data if requested and applicable
    :param data:
    :param do_sample:
    :param n_sample_row:
    :return:
    """
    if data is None or data.empty:
        return pd.DataFrame()

    if do_sample and len(data) > n_sample_row:
        return data.sample(n=n_sample_row, random_state=42)
    return data


def plot_univariate(data_to_plot, col, chart_type, ax):
    """
    :param data_to_plot:
    :param col:
    :param chart_type:
    :param ax:
    :return: helper function to plot single univariate chart
    """
    try:
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
            rotation = 45 if data_to_plot[col].nunique() > 7 else 0
            top_n = 20
            # Ensure column exists before value_counts
            if col not in data_to_plot.columns:
                raise ValueError(f"Column '{col}' not found in data for plotting.")
            order = data_to_plot[col].value_counts().nlargest(top_n).index

            if chart_type == "Count Plot":
                sns.countplot(x=data_to_plot[col], ax=ax, order=order)
                title = f"Count Plot - {col}"
                if len(data_to_plot[col].value_counts()) > top_n: title += f" (Top {top_n})"
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=rotation)
            elif chart_type == "Pie Chart":
                value_counts = data_to_plot[col].value_counts()
                plot_data = value_counts.nlargest(top_n)
                title = f"Pie Chart - {col}"
                if len(value_counts) > top_n:
                    plot_data['Other'] = value_counts.nsmallest(len(value_counts) - top_n).sum()
                    title += f" (Top {top_n} + Other)"

                if plot_data.empty or plot_data.sum() == 0:
                    ax.text(0.5, 0.5, 'No data to plot', horizontalalignment='center', verticalalignment='center')
                else:
                    plot_data.plot.pie(autopct="%1.1f%%", ax=ax, startangle=90, counterclock=False,
                                       wedgeprops={'linewidth': 0})
                ax.set_ylabel('')
                ax.set_title(title)
            else:  # Default categorical
                sns.countplot(x=data_to_plot[col], ax=ax, order=order)
                title = f"Count Plot - {col}"
                if len(data_to_plot[col].value_counts()) > top_n: title += f" (Top {top_n})"
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=rotation)
    except Exception as e:
        # Don't use st.error here, raise the error to be caught by the calling function
        # This allows the page script to display the error in the UI context
        raise RuntimeError(f"Error plotting univariate for '{col}' ({chart_type}): {e}") from e


def univariate_visualization(data, default_sample_flag, default_sample_size):
    """
    handles univariate feature visualization UI and plotting.
    """
    st.subheader('📈 Univariate Analysis (Single Feature)')
    if data is None:
        st.warning("No data available for visualization")
        return

    #----sampling UI----
    use_sampling = st.checkbox("Use Sampling for Univariate Viz?", value=default_sample_flag,
                               key="univar_sample_cb_unique")
    sample_size = default_sample_size
    if use_sampling:
        max_samples = min(50000, len(data))
        default_sample = min(default_sample_size, max_samples)
        sample_size = st.number_input("Sample size(univariate):", min_value=min(100, max_samples),
                                      max_value=max_samples,
                                      value=default_sample, step=100, key="univar_sample_size_unique")
    data_to_plot = get_sampled_data(data, use_sampling, sample_size)
    if data_to_plot.empty and not data.empty:
        st.warning("Sampled data is empty.")
        return
    elif data_to_plot.empty:
        st.warning("No data to plot.")
        return

    all_cols = data_to_plot.columns.tolist()
    if not all_cols:
        st.warning("No column found in the data for visualization.")
        return

    # feature selection UI
    if 'univar_feature_index' not in st.session_state:
        st.session_state.univar_feature_index = 0
    if st.session_state.univar_feature_index >= len(all_cols):
        st.session_state.univar_feature_index = 0

    selected_index = st.selectbox(
        "Select Feature: ",
        options=range(len(all_cols)),
        format_func=lambda x: all_cols[x],
        index=st.session_state.univar_feature_index,
        key="univar_select_unique"
    )
    st.session_state.univar_feature_index = selected_index
    feature = all_cols[selected_index]

    if feature:
        if pd.api.types.is_numeric_dtype(data_to_plot[feature]):
            default_chart = "Histogram"
            chart_options = ["Histogram", "Box Plot", "KDE Plot"]
        else:
            default_chart = "Count Plot"
            chart_options = ["Count Plot", "Pie Chart"]
        chart_key = f"univar_chart_type_{feature}"
        st.session_state.setdefault(chart_key, default_chart)
        try:
            current_chart_index = chart_options.index(st.session_state[chart_key])
        except ValueError:
            current_chart_index = chart_options.index(default_chart)
        selected_chart_type_index = st.selectbox("Choose Chart Type:", options=range(len(chart_options)),
                                                 format_func=lambda x: chart_options[x], index=current_chart_index,
                                                 key=f"univar_chart_select_{feature}_unique")
        chart_type = chart_options[selected_chart_type_index]
        st.session_state[chart_key] = chart_type

        # ----plotting--
        st.write(f"Generating **{chart_type}** for **{feature}**..")
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            plot_univariate(data_to_plot, feature, chart_type, ax)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"Could not generate plot for '{feature}': {ex}")
        finally:
            plt.close(fig)
            gc.collect()


def bivariate_visualization(data, target, default_sample_flag, default_sample_size):
    st.subheader('📉 Bivariate Analysis (Two Features)')
    if data is None or data.empty:
        st.warning("No data available.")
        return
    use_sampling = st.checkbox("Use Sampling ?", value=default_sample_flag, key="bivar_sample_cb_page")
    sample_size = default_sample_size
    if use_sampling:
        max_samples = min(50000, len(data))
        default_sample = min(default_sample_size, max_samples)
        sample_size = st.number_input("Sample size(bivariate):", min_value=min(100, max_samples),
                                      max_value=max_samples,
                                      value=default_sample, step=100, key="bivar_sample_size_unique")
        data_to_plot = get_sampled_data(data, use_sampling, sample_size)
        if data_to_plot.empty:
            st.warning("No data to plot.")
            return
        num_cols = data_to_plot.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data_to_plot.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = data_to_plot.columns.tolist()

        if len(all_cols) < 2:
            st.warning("number of features must greater that 2 ")
            return
        st.session_state.setdefault('bivar_feature_x_index', 0)
        st.session_state.setdefault('bivar_feature_y_index', 1 if len(all_cols) > 1 else 0)

        col1, col2 = st.columns(2)
        with col1:
            selected_x_index = st.selectbox("X-Axis:", options=range(len(all_cols)),
                                            format_func=lambda x: all_cols[x],
                                            index=st.session_state.bivar_feature_x_index,
                                            key="bivar_select_x_page")
            st.session_state.bivar_feature_x_index = selected_x_index
            feature_x = all_cols[selected_x_index]

        with col2:
            if selected_x_index == st.session_state.bivar_feature_y_index and len(all_cols) > 1:
                st.session_state.bivar_feature_y_index = 1 if selected_x_index == 0 else 0
            selected_y_index = st.selectbox("Y-Axis:", options=range(len(all_cols)), format_func=lambda x: all_cols[x],
                                            index=st.session_state.bivar_feature_y_index, key="bivar_select_y_page")
        st.session_state.bivar_feature_y_index = selected_y_index
        feature_y = all_cols[selected_y_index]

        if feature_x == feature_y:
            st.warning("Selected different features.")
            return

        st.write(f"Analyzing relationship between **{feature_x}** (X) and **{feature_y}** (Y)")
        x_is_num = feature_x in num_cols
        y_is_num = feature_y in num_cols
        #plot type UI and plotting call
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
                    sns.barplot(data=data_to_plot, x=feature_y, y=feature_x, ax=ax, order=order,
                                ci=None)  # ci=None faster
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
                    sns.barplot(data=data_to_plot, x=feature_x, y=feature_y, ax=ax, order=order,
                                ci=None)  # ci=None faster
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
                    sns.countplot(data=data_to_plot, x=feature_x, hue=feature_y, ax=ax, order=x_order,
                                  hue_order=hue_order)
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


def multivariate_visualization(data, target, default_sample_flag, default_sample_size):
    """
    handles multivariate feature visualization UI and plotting.
    """
    st.subheader("🧬 Multivariate Analysis (Multiple Features)")
    if data is None or data.empty:
        st.warning("No data available.")
        return
    use_sampling = st.checkbox("Use Sampling ?", value=default_sample_flag, key="multivar_sample_cb_page")
    sample_size = default_sample_size
    if use_sampling:
        max_samples = min(50000, len(data))
        default_sample = min(default_sample_size, max_samples)
        sample_size = st.number_input("Sample size(Multivariate):", min_value=min(100, max_samples),
                                      max_value=max_samples,
                                      value=default_sample, step=100, key="multivar_sample_size_unique")
        data_to_plot = get_sampled_data(data, use_sampling, sample_size)
        if data_to_plot.empty:
            st.warning("No data to plot.")
            return

        #plot type selection
        num_cols = data_to_plot.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least two numerical columns for standard multivariate plots like Pairplot or Heatmap.")
            return
        chart_options = ["Pairplot", "Correlation Heatmap"]
        st.session_state.setdefault("multivar_chart_type", 'pairplot')
        try:
            chart_index = chart_options.index(st.session_state.multivar_chart_type)
        except ValueError:
            chart_index = 0
        selected_chart_index = st.selectbox("Chart Type:", options=range(len(chart_options)),
                                            format_func=lambda x: chart_options[x], index=chart_index,
                                            key="multivar_chart_select_page")
        chart_type = chart_options[selected_chart_index]
        st.session_state.multivar_chart_type = chart_type

        # plotting logic
        if chart_type == "Pairplot":
            st.write(
                f"Generating Pairplot for numerical features (using target '{target}' for coloring if suitable)...")
            st.info("Pairplots can be computationally intensive for many features or large datasets.")
            pairplot_key = "multivar_pairplot_cols_state_page"
            default_cols = num_cols[:min(len(num_cols), 5)]
            st.session_state.setdefault(pairplot_key, default_cols)
            valid_defaults = [col for col in st.session_state[pairplot_key] if col in num_cols] or default_cols
            selected_num_cols = st.multiselect("Select numerical features for Pairplot:", num_cols,
                                               default=valid_defaults, key=pairplot_key)  # Default to first 5

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
                cols_for_pairplot = selected_num_cols + [
                    hue_col] if hue_col not in selected_num_cols else selected_num_cols
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
            correlation_heatmap_analysis(data_to_plot, target)


def correlation_heatmap_analysis(data, target=None):
    """
    calculates and display a correlation heatmap for numerical features.
    """
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
    sort_by_target = False
    if target and target in corr_matrix.columns:
        try:
            if num_data.isnull().sum().sum() > 0:
                imputer = SimpleImputer(strategy='median')
                num_data = pd.DataFrame(imputer.fit_transform(num_data), columns=num_data.columns,
                                        index=num_data.index)
            corr_matrix = num_data.corr()
        except Exception as ex:
            st.error(f"Correlation matrix error: {ex}")
            return
        sorted_key = "heatmap_sort_checkbox_page"
        st.session_state.setdefault(sorted_key, False)
        sort_by_target = st.checkbox(f"Sort heatmap by abs correlation with '{target}'?",
                                     value=st.session_state[sorted_key],
                                     key=sorted_key)
        if sort_by_target:
            try:
                target_corr = corr_matrix[target].abs().sort_values(ascending=False)
                corr_matrix = corr_matrix.loc[target_corr.index, target_corr.index]
            except Exception as ex:
                st.error(f"Error sorting heatmap: {ex}")
        fig, ax = plt.subplots(
            # This figsize calculation is functionally correct, though linters might warn
            figsize=(int(max(8.0, len(num_data.columns) * 0.6)), int(max(6.0, len(num_data.columns) * 0.5)))
        )
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        try:
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                        linewidths=.5, linecolor='lightgray', ax=ax, cbar=True, annot_kws={"size": 8})
            plt.title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"correlation matrix error: {ex}")
            return
        finally:
            plt.close(fig)
            gc.collect()
    else:
        try:
            if num_data.isnull().sum().sum() > 0:
                imputer = SimpleImputer(strategy='median')
                num_data = pd.DataFrame(imputer.fit_transform(num_data), columns=num_data.columns,
                                        index=num_data.index)
            corr_matrix = num_data.corr()
        except Exception as ex:
            st.error(f"Correlation matrix error: {ex}")
            return
        fig, ax = plt.subplots(
            # This figsize calculation is functionally correct, though linters might warn
            figsize=(int(max(8.0, len(num_data.columns) * 0.6)), int(max(6.0, len(num_data.columns) * 0.5)))
        )
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        try:
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                        linewidths=.5, linecolor='lightgray', ax=ax, cbar=True, annot_kws={"size": 8})
            plt.title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"correlation matrix error: {ex}")
            return
        finally:
            plt.close(fig)
            gc.collect()
