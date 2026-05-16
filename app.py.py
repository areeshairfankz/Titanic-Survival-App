import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to know if they would survive")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()

# Input form
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 30)
fare = st.sidebar.number_input("Fare ($)", min_value=0.0, value=50.0)

# Convert gender to number
sex_num = 1 if sex == "Male" else 0

# Predict button
if st.sidebar.button("Predict Survival"):
    # Create input dataframe
    input_data = pd.DataFrame([[pclass, sex_num, age, fare]], 
                              columns=['Pclass', 'Sex', 'Age', 'Fare'])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Show result
    if prediction == 1:
        st.success("✅ PASSENGER WOULD HAVE SURVIVED!")
        st.balloons()
    else:
        st.error("❌ PASSENGER WOULD NOT HAVE SURVIVED")