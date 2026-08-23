# 📐 Student Math Score Predictor

An end-to-end **Machine Learning project** that predicts students' math scores using academic and demographic information. The project includes data preprocessing, EDA, model training, evaluation, and an interactive Streamlit web application.

## 🚀 Features

- 📊 Exploratory Data Analysis using Matplotlib & Seaborn
- 🧹 Data preprocessing and feature engineering
- 🤖 Multiple regression models
- 📈 Model evaluation and comparison
- 🎯 Interactive math score prediction
- 🎲 Random student prediction
- 📊 Interactive Plotly visualizations
- 🌐 Streamlit web application

## 🤖 Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree
- Random Forest
- AdaBoost

## 🛠️ Tech Stack

**Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Plotly • Streamlit**

## 📂 Project Structure

```text
student-math-score-predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── stud.csv
│
└── src/
    ├── model_train.py
    └── preprocess_eda.py
    
▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

Then open:

http://localhost:8501
🌐 Deployment

The app can be deployed using Streamlit Community Cloud directly from GitHub.

Author: Usama Khokhar