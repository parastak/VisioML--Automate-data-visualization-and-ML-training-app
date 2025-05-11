# VisioML: Automated Data Visualization & ML Training App 🚀
Welcome to **VisioML** — your one-stop solution to automate the entire machine learning pipeline from raw data upload to model evaluation, without writing a single line of code.
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_CLOUD_APP_LINK_HERE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](YOUR_HUGGING_FACE_SPACE_LINK_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
VisioML is a comprehensive Streamlit application designed to empower data scientists and machine learning engineers by automating key stages of the ML workflow.
From data ingestion and cleaning to exploratory data analysis, feature engineering, model training, and evaluation, VisioML provides an intuitive interface to accelerate your projects.

## ✨ Features

VisioML offers a rich set of features to streamline your data science tasks:

* **⬆️ Data Ingestion:** Upload CSV or Excel files with ease. Handles multiple Excel sheets.
* **🧹 Automated Data Cleaning:**
    * Identifies and handles missing values (imputation or dropping).
    * Detects and removes duplicate rows.
    * Option to treat empty strings as missing values.
  
* **⚙️ Feature Engineering:**
    * **Categorical Encoding:** Supports One-Hot, Label (Ordinal), and Binary encoding.
    * **Numerical Scaling:** Offers MinMaxScaler, StandardScaler, and RobustScaler, with an auto-detect option.
      
* **📊 Interactive EDA & Visualization:**
    * **Data Quality Assessment:** Detailed overview of data types, missing values, duplicates, and descriptive statistics.
    * **Univariate Analysis:** Histograms, Box Plots, KDE Plots for numerical features; Count Plots, Pie Charts for categorical features.
    * **Bivariate Analysis:** Scatter plots, Line plots, Box plots (Num-Cat), Violin plots (Num-Cat), Grouped Bar charts (Cat-Cat), etc.
    * **Multivariate Analysis:** Pair plots and Correlation Heatmaps.
    * Sampling options for large datasets.
    
* **🎯 Target Definition & Problem Typing:** Select your target variable and automatically determine if it's a Regression, Binary Classification, or Multiclass Classification problem.
* **⭐ Feature Importance & Selection:**
    * Calculates feature importance scores (F-Score, Mutual Information, Pearson Correlation).
    * Allows manual selection of features for model training.
  
* **🤖 Machine Learning Modeling:**
    * **Data Splitting:** Customizable test set size, random state, and stratification.
    * **Model Selection:** Supports a wide range of Scikit-learn models for both classification and regression tasks (e.g., Logistic/Linear Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Trees, Naive Bayes).
    * **Cross-Validation:** Perform K-Fold or Stratified K-Fold cross-validation on the training set.
    * **Model Training:** Train selected models on the training data (full or sample).
    * **Comprehensive Evaluation:**
        * Detailed metrics for classification (Accuracy, Precision, Recall, F1-score, ROC AUC) and regression (MAE, MSE, RMSE, R²).
        * Visualizations: Confusion Matrix, ROC Curve (for classification), Actual vs. Predicted plots (for regression).
        * Classification reports.
    * **Model Download:** Download trained models as `.joblib` files.
* **📄 Modular & User-Friendly:** Built with a modular codebase (separate utility files for logic and page files for UI) for maintainability and a clean, intuitive Streamlit interface.

---

## 🛠️ Tech Stack

| Tool       | Description                            |
|------------|----------------------------------------|
| `Python`   | Core programming language              |
| `Streamlit`| Web UI for ML workflow                 |
| `Scikit-learn` | ML algorithms and metrics          |
| `Pandas/Numpy`| Data handling and transformation    |
| `Matplotlib/Seaborn` | Data visualization           |

---

## 🚀 Live Demo

Experience VisioML in action!

* **Streamlit Community Cloud:** [YOUR_STREAMLIT_CLOUD_APP_LINK_HERE](YOUR_STREAMLIT_CLOUD_APP_LINK_HERE)
* **Hugging Face Spaces:** [YOUR_HUGGING_FACE_SPACE_LINK_HERE](YOUR_HUGGING_FACE_SPACE_LINK_HERE)

*(Please be patient if the app is waking up from sleep on the free tiers.)*

## 📸 Screenshots

*(Add 2-3 compelling screenshots of your app here. You can drag-and-drop images directly into the GitHub text editor for the README, or upload them to your repo (e.g., in an `assets/images/` folder) and link them like this: `![Screenshot 1](assets/images/screenshot1.png)`)*

**Example Screenshot Placeholder:**
![VisioML Interface](assets/images/placeholder_screenshot1.png)
*Caption: The main data loading and navigation interface.*

![VisioML Visualization](assets/images/placeholder_screenshot2.png)
*Caption: Example of the interactive visualization page.*

## 🎞️ Demo Video / GIF

*(A short GIF is highly recommended! You can use tools like ScreenToGif (Windows), Kap (Mac), or Peek (Linux) to record a short walkthrough. Upload the GIF to your repo (e.g., `assets/demo.gif`) and embed it.)*

**Example GIF Placeholder:**
![VisioML Workflow Demo](assets/demo.gif)


## ⚙️ Setup and Run Locally
To run VisioML on your local machine, follow these steps:

1.  **Prerequisites:**
    * Python 3.8 - 3.11
    * pip (Python package installer)
    * Git

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/parastak/VisioML--Automate-data-visualization-and-ML-training-app.git](https://github.com/parastak/VisioML--Automate-data-visualization-and-ML-training-app.git)
    cd VisioML--Automate-data-visualization-and-ML-training-app
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    * **Windows:**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app2.py
    ```
    The application should open in your default web browser.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/parastak/VisioML--Automate-data-visualization-and-ML-training-app/issues) if you want to contribute.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
