import streamlit as st
import pandas as pd
import pickle



def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# OG models
xgb_model = load_model("../models/xgb_model.pkl")
naive_bayes_model = load_model("../models/nb_model.pkl")
rf_model = load_model("../models/rf_model.pkl")
svm_model = load_model("../models/svm_model.pkl")
decision_tree_model = load_model("../models/dt_model.pkl")
knn_model = load_model("../models/knn_model.pkl")

# middle models
xgboost_SMOTE_model = load_model("../models/xgbboost-SMOTE.pkl")
xgboost_featureEngineered_model = load_model("../models/xgbboost-featureEngineered.pkl")
voting_classifier_model = load_model("../models/voting_clf.pkl")


# Ensemble models
soft_voting_model = load_model("../models/voting_clf.pkl")
weighted_voting_model = load_model("../models/weighted_voting_tuned.pkl")
best_ensemble_model = load_model("../models/best_ensemble_model.pkl")



def prepare_input(credit_score, location, gender, age, tenure, balance, num_products,
    is_active_member, estimated_salary):

    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": is_active_member,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geograph_France": 1 if location == "France" else 0,
        "Geograph_Germany": 1 if location == "Germany" else 0,
        "Geograph_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    
    return input_df, input_dict




st.title("Customer Churn Prediction")

df = pd.read_csv("./data/churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])

    selected_surname = selected_customer_option.split(" - ")[1]

    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        
        credit_score = st.number_input("Credit Score", value=int(selected_customer['CreditScore']), min_value=350, max_value=850)

        location = st.selectbox("Location", options=["France", "Germany", "Spain"], index=selected_customer['Geography'].index(selected_customer['Geography']))

        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == "Male" else 1)

        age = st.number_input("Age", value=int(selected_customer['Age']), min_value=18, max_value=100, step=1)

        tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, step=1, value=int(selected_customer['Tenure']))
    
    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer['Balance']))

        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts']))

        has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard']))

        is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer['EstimatedSalary']))

        
    

