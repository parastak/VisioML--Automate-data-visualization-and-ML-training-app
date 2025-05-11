import pandas as pd
import streamlit as st
import numpy as np
import gc


def downcast_dtypes(df):
    """Reduces memory usage by downcasting numerical dtypes."""
    st.write("📉 Attempting to downcast numerical dtypes for memory optimization...")
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.select_dtypes(include=['int', 'float']).columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(col_type):
            # Skip downcasting floats for now, can sometimes lose precision
            # Can add float16/float32 logic here if needed and acceptable
            pass  # Example: df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    st.info(
        f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({((start_mem - end_mem) / start_mem) * 100:.1f}% reduction)")
    gc.collect()  # Explicitly collect garbage
    return df


# ------------------------------ #
#   File Upload & Dataset Loader   #
# ------------------------------ #


def load_dataset(uploaded_file):
    try:
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            st.info(f"Loading `{uploaded_file.name}` ({file_extension})...")

            if file_extension == "csv":
                data = pd.read_csv(uploaded_file)
                msg = "✅ CSV file loaded successfully."
                sheet_names = None

            elif file_extension == "xlsx":
                # Read all sheets as dict
                sheet_dict = pd.read_excel(uploaded_file, sheet_name=None)
                sheet_names = list(sheet_dict.keys())

                selected_sheet = st.selectbox("Select a sheet to load:", sheet_names)
                data = sheet_dict[selected_sheet]
                msg = f"✅ Excel sheet '{selected_sheet}' loaded successfully"
            else:
                st.error("unsupported file type . please upload a CSV or Excel file.")
                return None, "Error: unsupported file type. Please upload CSV or Excel.", None

            if data is not None and data.empty:
                st.warning("the uploaded file is empty.")
                return None, "Warn: The file is empty.", sheet_names

            elif data is not None:
                st.success("✅ File loaded successfully!")
                st.write("📊 **Preview of the uploaded dataset:**")
                st.dataframe(data.head(15))
                data = downcast_dtypes(data.copy())  # downcast after loading
                return data, msg, sheet_names
        else:
            return None, "Info: No file uploaded.", None

    except Exception as ex:
        st.error(f"an error occurred while loading the file : {ex}")
        return None, f"Error loading file: {ex}", None
