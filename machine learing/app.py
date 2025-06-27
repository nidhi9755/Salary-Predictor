
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Salary Predictor", layout="centered")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Prediction"])

# Load and clean data
df = pd.read_csv("Salary_dataset.csv")
df.drop(columns='Unnamed: 0', inplace=True)

# Page 1: Introduction
if page == "Introduction":
    st.title(" Salary Predictor App")
    st.markdown("""
    ##  Dataset Information
    This dataset contains information about salaries and years of experience of employees.
    
    **Columns:**
    - `YearsExperience`: Number of years a person has worked.
    - `Salary`: Annual salary in INR.

    ##  Purpose of the App
    - Perform **Exploratory Data Analysis (EDA)**.
    - Predict **Salary** based on **Years of Experience** using **Linear Regression**.
    """)

# Page 2: EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Visualizations")

    if st.button("Show Correlation Heatmap"):
        fig1, ax1 = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)

    if st.button("Show Scatter Plot"):
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='YearsExperience', y='Salary', data=df, ax=ax2)
        st.pyplot(fig2)

    if st.button("Show Line Plot"):
        fig3, ax3 = plt.subplots()
        sns.lineplot(x='YearsExperience', y='Salary', data=df, marker='o', ax=ax3)
        st.pyplot(fig3)

    if st.button("Show Box Plot"):
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df[['YearsExperience', 'Salary']], ax=ax4)
        st.pyplot(fig4)

# Page 3: Prediction
else:
    st.title("Salary Prediction using Linear Regression")

    X = df[['YearsExperience']]
    y = df['Salary']
    model = LinearRegression()
    model.fit(X, y)

    st.write("### Enter Years of Experience")
    years = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

    if st.button("Predict Salary"):
        prediction = model.predict([[years]])
        st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")