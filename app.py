import pandas as pd
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")

import logging
import warnings

warnings.filterwarnings("ignore")

# Suppress Streamlit runtime warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

# Alternatively, still use warnings to filter UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")




# Load the dataset
file_path = r"C:\Users\HP\loan_prediction_app\data\cleaned_loan_data.csv"
df = pd.read_csv(file_path)


# Preview the data
print(df.head())


# Check data types and missing values
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Define the target column
target_column = 'Loan_Status'

# Drop Loan_ID columns and the target column from features
X = df.drop(columns=[target_column] + [col for col in df.columns if col.startswith('Loan_ID')])

# Define the target variable
y = df[target_column]

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Check for missing values again
print(X.isnull().sum())

# Initialize the scaler
scaler = StandardScaler()

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of numeric columns to scale
numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Fit the scaler on training data
scaler.fit(X_train[numeric_columns])

# Now scale the training and test data
X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])


# Initialize the model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')


# Print classification report

st.write(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write('Confusion Matrix:')
st.write(conf_matrix)

model_path = r"C:\Users\HP\loan_prediction_app\model.pkl"
scaler_path = r"C:\Users\HP\loan_prediction_app\scaler.pkl"

# Save the trained model to a file
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Scaler saved at {scaler_path}")

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit input fields for new data
ApplicantIncome = st.number_input("Applicant's Income")
CoapplicantIncome = st.number_input("Coapplicant's Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Amount Term")
Credit_History = st.number_input("Credit History (1 or 0)")
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Gender = st.selectbox("Gender", ["Male", "Female"])

# Create a DataFrame from user inputs
sample_data = pd.DataFrame({
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Dependents': [Dependents],
    'Education': [Education],
    'Gender': [Gender]
   
})

# Ensure sample data has the same columns as the training data (dummy variables)
# You might need to apply one-hot encoding or similar transformations for categorical variables
sample_data = pd.get_dummies(sample_data)

# Match columns with the model input data
missing_cols = set(X.columns) - set(sample_data.columns)
for col in missing_cols:
    sample_data[col] = 0

# Reorder columns to match the model input
sample_data = sample_data[X.columns]

# Scale the data using the pre-fitted scaler (only for numeric columns)
sample_data[numeric_columns] = scaler.transform(sample_data[numeric_columns])

# Make prediction when the button is pressed
if st.button('Predict Loan Status'):
    prediction = model.predict(sample_data)  # Make prediction using the logistic regression model
    if prediction[0] == 1:
        st.write("Loan Approved!")
    else:
        st.write("Loan Denied!")



