import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Streamlit Title
st.title('Banking App Predictive Modeling')

# File Uploaders
train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

if train_file is not None:
    # Load training data
    df_train = pd.read_csv(train_file)
    st.write("Training Data Preview", df_train.head())

    # Data Preprocessing
    categorical_cols = df_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_train[col] = df_train[col].replace('unknown', df_train[col].mode()[0])

    # Encoding categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        df_train[col] = le.fit_transform(df_train[col])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

    # Split the data
    X_train = df_train.drop('y', axis=1)
    y_train = df_train['y']
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and Evaluation on Training Data
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate R² and RMSE for Training Data
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    st.write(f"Training Data - R²: {r2_train:.2f}, RMSE: {rmse_train:.2f}")

    # Model evaluation for validation set
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    st.write(f"Validation Data - R²: {r2_val:.2f}, RMSE: {rmse_val:.2f}")

    # Optionally plot some graphs for training data analysis
    plt.figure(figsize=(8, 6))
    sns.countplot(x='y', data=df_train)
    st.pyplot(plt)

if test_file is not None:
    # Load test data
    df_test = pd.read_csv(test_file)
    st.write("Test Data Preview", df_test.head())

    # Data Preprocessing on Test Data (similar to the training data)
    for col in categorical_cols:
        df_test[col] = df_test[col].replace('unknown', df_test[col].mode()[0])

    # Encoding categorical features
    for col in categorical_cols:
        df_test[col] = le.transform(df_test[col])

    # Scale numerical features
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

    # Split the test data
    X_test = df_test.drop('y', axis=1)
    y_test = df_test['y']

    # Predictions and Evaluation on Test Data
    y_test_pred = model.predict(X_test)

    # Calculate R² and RMSE for Test Data
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    st.write(f"Test Data - R²: {r2_test:.2f}, RMSE: {rmse_test:.2f}")