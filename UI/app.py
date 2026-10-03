import os
import glob
import streamlit as st
import pandas as pd
import joblib

from output_and_scoring import render_output_browser, render_scoring
# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Prediction",
    page_icon="💳",
    layout="centered"
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
# Build paths relative to this script's own location, rather than
# the current working directory. This way the app works the same
# whether it's run from Spyder, VS Code, a terminal in a different
# folder, or Streamlit Community Cloud.
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "logistic_regression_model.pkl"
# app.py lives in UI/, and the model lives in ../model/ relative to it
MODEL_PATH = os.path.join(APP_DIR, "..", "model", MODEL_FILENAME)

# app.py lives in UI/, and saved plots/CSVs live in ../output/plots/
PLOTS_DIR = os.path.join(APP_DIR, "..", "output", "plots")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: `{MODEL_FILENAME}`.\n\n"
            "Make sure this file is committed to the same folder as `app.py` "
            "in your GitHub repository."
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(
            "The model file couldn't be loaded. This usually means the "
            "scikit-learn version installed here doesn't match the version "
            "used to train the model. Check `requirements.txt` and pin "
            "scikit-learn to match.\n\n"
            f"Details: {e}"
        )
        st.stop()


model = load_model()


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("💳 Credit Card Prediction")


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab_outputs, tab_scoring = st.tabs([
    "Data Exploration",
    "Logistic Regression",
    "Random Forest",
    "Model Comparison",
    "Model Outputs",
    "Score New File",
])



#tab_predict, tab_explore, tab_outputs, tab_scoring = st.tabs(
 #   ["🔮 Prediction", "📊 Data Exploration", "📁 Model Outputs", "🆕 Score New File"]
#)


with tab_predict:

    st.write(
        "Enter the customer information below to predict "
        "the likelihood of being a credit card holder."
    )
    with tab_outputs:
    render_output_browser()

    with tab_scoring:
         render_scoring()

    with tab_outputs:
         render_output_browser()

    with tab_scoring:
    render_scoring()
    
    # -------------------------------------------------------
    # Customer inputs
    # -------------------------------------------------------

    st.subheader("Customer Information")


    retirement_balance = st.number_input(
        "Retirement Balance ($)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    brokerage_balance = st.number_input(
        "Brokerage Balance ($)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    credit_utilization_percent = st.number_input(
        "Credit Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    autopay_enrolled = st.selectbox(
        "Autopay Enrolled",
        ["Yes", "No"]
    )

    overdraft_count = st.number_input(
        "Overdrafts (12 months)",
        min_value=0,
        value=0,
        step=1
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    risk_tier = st.selectbox(
        "Risk Tier",
        ["Prime Plus", "Prime", "Other"]
    )

    delinquency_90d_count = st.number_input(
        "90-Day Delinquencies",
        min_value=0,
        value=0,
        step=1
    )

    # -------------------------------------------------------
    # Prediction
    # -------------------------------------------------------

    st.divider()

    if st.button("🔮 Predict", type="primary"):

        # Convert Streamlit inputs to model variables
        autopay_flag = 1 if autopay_enrolled == "Yes" else 0

        # Convert percentage to ratio
        credit_utilization_ratio = credit_utilization_percent / 100

        # One-hot encoding for risk tier
        risk_tier_prime_plus = 1 if risk_tier == "Prime Plus" else 0
        risk_tier_prime = 1 if risk_tier == "Prime" else 0

        # Create input dataframe
        input_data = pd.DataFrame({
            "retirement_balance": [retirement_balance],
            "brokerage_balance": [brokerage_balance],
            "credit_utilization_ratio": [credit_utilization_ratio],
            "autopay_enrolled_flag": [autopay_flag],
            "overdraft_count_12mo": [overdraft_count],
            "age": [age],
            "risk_tier_Prime Plus": [risk_tier_prime_plus],
            "risk_tier_Prime": [risk_tier_prime],
            "delinquency_90d_count": [delinquency_90d_count]
        })

        # Make prediction
        probability = model.predict_proba(input_data)[0, 1]
        prediction = model.predict(input_data)[0]

        # -----------------------------------------------------
        # Display results
        # -----------------------------------------------------

        st.subheader("Prediction Result")

        probability_percent = probability * 100

        st.metric(
            "Prediction Probability",
            f"{probability_percent:.1f}%"
        )

        if prediction == 1:
            st.success("🟢 Likely Credit Card Holder")
        else:
            st.warning("🔴 Unlikely Credit Card Holder")

        # Show probability bar
        st.progress(float(probability))

        # Show technical details
        with st.expander("View model inputs"):
            st.dataframe(input_data)


with tab_explore:

    st.write(
        "Saved data exploration and correlation output from the "
        "modeling pipeline."
    )

    if not os.path.isdir(PLOTS_DIR):
        st.info(
            "No `output/plots` folder found yet. Once you save exploration "
            "images or CSVs there and push to GitHub, they'll show up here "
            "automatically."
        )
    else:
        # Show every image in the folder (png, jpg, jpeg), sorted by name
        image_paths = sorted(
            glob.glob(os.path.join(PLOTS_DIR, "*.png"))
            + glob.glob(os.path.join(PLOTS_DIR, "*.jpg"))
            + glob.glob(os.path.join(PLOTS_DIR, "*.jpeg"))
        )

        # Show every CSV in the folder, sorted by name
        csv_paths = sorted(glob.glob(os.path.join(PLOTS_DIR, "*.csv")))

        if not image_paths and not csv_paths:
            st.info("The `output/plots` folder is empty.")

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            title = filename.rsplit(".", 1)[0].replace("_", " ").title()
            st.subheader(title)
            st.image(image_path, use_container_width=True)
            st.divider()

        for csv_path in csv_paths:
            filename = os.path.basename(csv_path)
            title = filename.rsplit(".", 1)[0].replace("_", " ").title()
            st.subheader(title)
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't read `{filename}`: {e}")
            st.divider()

            with tab_outputs:
                 render_output_browser()

            with tab_scoring:
                 render_scoring()
