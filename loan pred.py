# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:09:40 2025

@author: SSD
"""
import sklearn
import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("loan_status_model.pkl", "rb") as file:
    classifier = pickle.load(file)

st.title("🏦 Loan Approval Prediction Web App By Ayman")

st.markdown("### Enter applicant details below to predict loan status:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.text_input("Applicant Income (e.g., 5000)", "0")
coapplicant_income = st.text_input("Coapplicant Income (e.g., 2000)", "0")
loan_amount = st.text_input("Loan Amount (in thousands, e.g., 128)", "100")
loan_term = st.text_input("Loan Term (in months, e.g., 360)", "360")

credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert text inputs to integers
try:
    applicant_income = int(applicant_income)
    coapplicant_income = int(coapplicant_income)
    loan_amount = int(loan_amount)
    loan_term = int(loan_term)
except ValueError:
    st.error("❌ Please enter valid numeric values for income, loan amount, and loan term.")
    st.stop()

# Encoding categorical variables manually
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 4 if dependents == "3+" else int(dependents)
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Create input array
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_term, credit_history, property_area]])

# Prediction
if st.button("Predict Loan Status"):
    prediction = classifier.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan is likely to be *Approved*.")
    else:
        st.error("❌ Loan is likely to be *Rejected*.")
