import streamlit as st
import pandas as pd
import numpy as np
import joblib
 # If you're using a different method to save/load your model, adjust this import

# Function to load the trained model
def load_model():
    # Replace 'your_model_path.pkl' with the actual path to your trained model file
    model = joblib.load('/home/chenxi/data/DSC148/final/model_filename.pkl')
    return model

# Setup the app layout
st.title('League of Legends Match Predictor')

# Create input fields for all features required by the model
goldat10 = st.number_input('Gold at 10 minutes', min_value=0)
firstblood = st.selectbox('First Blood', options=['0', '1'])
firstdragon = st.selectbox('First Dragon', options=['0', '1'])
firsttower = st.selectbox('First Tower', options=['0', '1'])
xpat15 = st.number_input('XP at 15 minutes', min_value=0)
goldat15 = st.number_input('Gold at 15 minutes', min_value=0)
firstmidtower = st.selectbox('First Mid Tower', options=['0', '1'])
kda_15 = st.number_input('KDA at 15 minutes', min_value=0.0, format='%f')
league = st.text_input('League')

# Button to make prediction
if st.button('Predict Match Outcome'):
    # Load the model
    model = load_model()
    
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'goldat10': [goldat10],
        'firstblood': [firstblood],
        'firstdragon': [firstdragon],
        'firsttower': [firsttower],
        'xpat15': [xpat15],
        'goldat15': [goldat15],
        'firstmidtower': [firstmidtower],
        'kda_15': [kda_15],
        'league': [league]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    result = 'Win' if prediction[0] == 1 else 'Lose'
    st.success(f'The predicted outcome is: {result}')

