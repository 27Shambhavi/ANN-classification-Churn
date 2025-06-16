import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit_Score')
estimated_salary = st.number_input('Estimated_Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Prepare numeric features
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode geography
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Combine all features
full_input = pd.concat([geo_encoded_df, input_data], axis=1)

# Reorder columns to match training
full_input = full_input[scaler.feature_names_in_]

# Scale
input_data_scaled = scaler.transform(full_input)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display
if prediction_proba > 0.5:
    st.error(f"The customer is likely to churn. (Probability: {prediction_proba:.2f})")
else:
    st.success(f"The customer is not likely to churn. (Probability: {prediction_proba:.2f})")
