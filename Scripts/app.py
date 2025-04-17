import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd

# Load the encoder and model
ohe = pk.load(open('../Artifacts/encoding.pkl', 'rb'))
linear_regression = pk.load(open('../Artifacts/lr_log.pkl', 'rb'))

st.title("Predict Insurance Charges")

# Actual age input
age_val = st.number_input(label='Age', min_value=18, max_value=64, value=30)
bmi = st.number_input(label="BMI", min_value=10.0, max_value=60.0)
children = st.number_input(label="Children", min_value=0, max_value=5)
smoke = st.selectbox(label="Smoke", options=("yes", "no"))
sex = st.selectbox(label="Sex", options=("male", "female"))
region = st.selectbox(label="Region", options=("southeast", "southwest", "northwest", "northeast"))

# Define age bins for encoding
age_bins = [0, 18, 24, 34, 44, 54, 64, float('inf')]
age_labels = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 and above']
age_cat = pd.cut([age_val], bins=age_bins, labels=age_labels)[0]

# Only use categories that exist in your encoder
# Transform categorical features
encoded = ohe.transform([[sex, smoke, region, age_cat]]).toarray()

    # Combine numerical features with encoded features
final_input = np.concatenate([[[age_val, bmi, children]], encoded], axis=1)
if st.button("Predict Charges"):
    log_pred = linear_regression.predict(final_input)[0]
    pred_charges = np.exp(log_pred)  # if model used log1p
    st.success(f"Estimated Insurance Charges: ${pred_charges:,.2f}")
