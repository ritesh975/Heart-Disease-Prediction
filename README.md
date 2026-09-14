# Heart-Disease-Prediction
An ANN-based Heart Disease Prediction project built with Python, TensorFlow/Keras and Streamlit, featuring data preprocessing, feature scaling, neural network training, model evaluation, and an interactive web interface.
# ❤️ Heart Disease Prediction using Artificial Neural Network

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Keras](https://img.shields.io/badge/Keras-Neural%20Network-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## 📌 Overview

This project demonstrates the implementation of an **Artificial Neural Network (ANN)** for heart disease prediction using a structured heart disease dataset.

The complete workflow covers:

- Dataset loading and exploration
- Feature and target separation
- Train-test splitting
- Feature standardization
- Artificial Neural Network development
- Model training
- Regression-based evaluation
- Prediction analysis
- Interactive Streamlit web application

The project is developed for **educational and learning purposes**, with a focus on understanding the practical implementation of an ANN model and its deployment through a web interface.

---

## 🎯 Project Objectives

The primary objectives of this project are:

- To understand the working of Artificial Neural Networks.
- To preprocess and standardize medical-related numerical data.
- To build an ANN using TensorFlow/Keras.
- To understand hidden layers, neurons and activation functions.
- To implement a linear output layer.
- To evaluate model performance using regression metrics.
- To integrate the trained workflow with Streamlit.
- To develop an interactive prediction interface.

---

## 📊 Dataset

The project uses the `heart.csv` dataset.

### Dataset Information

- **Total Records:** 1,025
- **Total Columns:** 14
- **Input Features:** 13
- **Target Column:** `target`

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Cholesterol level |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting ECG result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `target` | Target value |

---

# 🧠 Artificial Neural Network Architecture

The ANN consists of two hidden layers followed by a single linear output neuron.

```text
                 Input Layer
                13 Features
                     │
                     ▼
            ┌─────────────────┐
            │ Dense Layer     │
            │ 16 Neurons      │
            │ ReLU Activation  │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Dense Layer     │
            │ 8 Neurons       │
            │ ReLU Activation  │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Output Layer    │
            │ 1 Neuron        │
            │ Linear          │
            └────────┬────────┘
                     │
                     ▼
                Prediction
