# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Suppress warnings for a cleaner UI
warnings.filterwarnings('ignore')

# --- 1. Data Loading and Preprocessing Simulation ---
def load_and_preprocess_data():
    """
    Loads and preprocesses the training data, returning the processed features,
    labels, and the fitted LabelEncoders.
    """
    try:
        data = pd.read_csv('final.csv', on_bad_lines='skip', low_memory=False)
    except FileNotFoundError:
        st.error("Error: 'final.csv' not found. Please ensure the file is in the same directory.")
        return None, None, None, None

    # Apply the feature engineering step from the notebook
    data['char'] = data['nameDest'].apply(lambda x: x[0])

    # Select the features used in the model
    selected_features_df = data[[
        'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'char', 'isFraud'
    ]].copy()

    # Initialize and fit LabelEncoders on the original string data
    le_type = LabelEncoder()
    selected_features_df['type'] = le_type.fit_transform(selected_features_df['type'])

    le_char = LabelEncoder()
    selected_features_df['char'] = le_char.fit_transform(selected_features_df['char'])

    # Separate features and labels
    features = selected_features_df.drop('isFraud', axis=1)
    labels = selected_features_df['isFraud']

    # Return the processed data and the fitted encoders
    return features, labels, le_type, le_char

def train_and_save_model(features, labels, le_type, le_char):
    """
    Simulates the model training and saving process.
    This function will be called once to prepare the model file.
    """
    # Split data into training and validation sets
    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.3, random_state=0
    )

    # Apply SMOTEENN for oversampling to handle class imbalance
    sme = SMOTEENN(random_state=42, sampling_strategy='auto')
    x_sme, y_sme = sme.fit_resample(x_train, y_train)

    # Use a Pipeline to scale the data and train the model.
    # This prevents ConvergenceWarnings for Logistic Regression and improves performance
    # for other models like SVM (though RF is not sensitive to scaling).
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=0))
    ])

    # Train the model pipeline
    model_pipeline.fit(x_sme, y_sme)

    # Save the trained model pipeline AND the encoders to a single file
    with open('final_model_and_encoders.pkl', 'wb') as file:
        pickle.dump({
            'model_pipeline': model_pipeline,
            'le_type': le_type,
            'le_char': le_char
        }, file)
    
    st.success("Model and encoders have been successfully trained and saved!")

    return model_pipeline, le_type, le_char

# --- Run the training process to create the model file ---
# This block runs on the first script execution to generate the .pkl file
# and is then skipped by Streamlit's caching.
features, labels, le_type_trained, le_char_trained = load_and_preprocess_data()
if features is not None:
    # This will create the 'final_model_and_encoders.pkl' file
    model, le_type, le_char = train_and_save_model(features, labels, le_type_trained, le_char_trained)

# --- 2. Streamlit Web App Interface ---

# Load the trained model and encoders from the pickled file
@st.cache_resource
def load_resources():
    try:
        # Load the dictionary containing the model pipeline and the encoders
        with open('final_model_and_encoders.pkl', 'rb') as file:
            saved_objects = pickle.load(file)
        
        return saved_objects['model_pipeline'], saved_objects['le_type'], saved_objects['le_char']
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'final.csv' is in the directory and run the app once to generate it.")
        return None, None, None

model_pipeline, le_type, le_char = load_resources()

if model_pipeline and le_type and le_char:
    st.title("PaySim Fraud Detection App")
    st.write("Enter a transaction's details to predict if it is fraudulent.")
    
    # User input fields
    st.header("Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
        )
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        old_balance_org = st.number_input("Original Balance (Sender)", min_value=0.0, format="%.2f")
        old_balance_dest = st.number_input("Original Balance (Receiver)", min_value=0.0, format="%.2f")
        
    with col2:
        new_balance_org = st.number_input("New Balance (Sender)", min_value=0.0, format="%.2f")
        new_balance_dest = st.number_input("New Balance (Receiver)", min_value=0.0, format="%.2f")
        dest_char_input = st.selectbox(
            "Destination Account Type",
            ('Customer (C)', 'Merchant (M)')
        )

    # Extract the single character for encoding
    dest_char_map = {'Customer (C)': 'C', 'Merchant (M)': 'M'}
    dest_char_single = dest_char_map[dest_char_input]

    # Preprocess user input for the model
    def preprocess_input(input_data):
        try:
            # Create a DataFrame from the single input
            input_df = pd.DataFrame([input_data], columns=[
                'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'char'
            ])

            # Apply the same LabelEncoders as used in training
            input_df['type'] = le_type.transform(input_df['type'])
            input_df['char'] = le_char.transform(input_df['char'])
            
            return input_df
        except ValueError as e:
            st.error(f"Error during preprocessing. Make sure input values are correct. Details: {e}")
            return None

    # Prediction button
    if st.button("Predict Fraud"):
        input_data = {
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_org,
            'newbalanceOrig': new_balance_org,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'char': dest_char_single
        }
        
        processed_input = preprocess_input(input_data)
        
        if processed_input is not None:
            # Make a prediction using the full pipeline
            prediction = model_pipeline.predict(processed_input)
            prediction_proba = model_pipeline.predict_proba(processed_input)
            
            st.header("Prediction Result")
            if prediction[0] == 1:
                st.error(f"This transaction is predicted as **FRAUDULENT** with a probability of {prediction_proba[0][1]:.2%}.")
            else:
                st.success(f"This transaction is predicted as **NOT FRAUDULENT** with a probability of {prediction_proba[0][0]:.2%}.")

            st.write("---")
            st.subheader("Model Input Summary (Encoded)")
            st.write(processed_input)