import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)



# LOAD MODEL AND SCALER


model = tf.keras.models.load_model("ann_model.keras")
scaler = joblib.load("scaler.pkl")


# TITLE


st.title("❤️ Heart Disease Prediction 🫀")

st.subheader("ANN Model with Linear Regression")

st.write(
    "Enter patient information and predict the target "
    "using an Artificial Neural Network."
)

st.divider()


# PATIENT INFORMATION


st.header("🧑 Patient Information")


col1, col2, col3 = st.columns(3)



# Column 1


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    cp = st.number_input(
        "Chest Pain Type (cp)",
        min_value=0,
        max_value=3,
        value=0
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )


# Column 2


with col2:

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=600,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar (fbs)",
        [0, 1]
    )

    restecg = st.number_input(
        "Resting ECG (restecg)",
        min_value=0,
        max_value=2,
        value=0
    )

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=50,
        max_value=250,
        value=150
    )



# Column 3

with col3:

    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.number_input(
        "Slope",
        min_value=0,
        max_value=2,
        value=1
    )

    ca = st.number_input(
        "Major Vessels (ca)",
        min_value=0,
        max_value=4,
        value=0
    )

    thal = st.number_input(
        "Thal",
        min_value=0,
        max_value=3,
        value=2
    )


st.divider()

# PREDICTION


if st.button("🔍 Predict", use_container_width=True):

    # Create input data
    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Create DataFrame
    input_df = pd.DataFrame(
        input_data,
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    result = float(prediction[0][0])
    
    # RESULT
    st.header("📊 Prediction Result")

    st.metric(
        "Predicted Target",
        f"{result:.4f}"
    )


    if result >= 0.5:

        st.error(
            "⚠️ Higher likelihood of heart disease"
        )

    else:

        st.success(
            "✅ Lower likelihood of heart disease"
        )


    st.info(
        "This prediction is for educational purposes "
        "and should not be used as medical advice."
    )


# FOOTER


st.divider()

st.caption(
    "ANN + Linear Regression | "
    "Python | TensorFlow | Streamlit"
)