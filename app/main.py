import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)





def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# OG models
xgb_model = load_model("./models/xgb_model.pkl")
naive_bayes_model = load_model("./models/nb_model.pkl")
rf_model = load_model("./models/rf_model.pkl")
svm_model = load_model("./models/svm_model.pkl")
decision_tree_model = load_model("./models/dt_model.pkl")
knn_model = load_model("./models/knn_model.pkl")

# middle models
xgboost_SMOTE_model = load_model("./models/xgbboost-SMOTE.pkl")
xgboost_featureEngineered_model = load_model("./models/xgbboost-featureEngineered.pkl")
voting_classifier_model = load_model("./models/voting_clf.pkl")


# Ensemble models
soft_voting_model = load_model("./models/voting_clf.pkl")
weighted_voting_model = load_model("./models/weighted_voting_tuned.pkl")
best_ensemble_model = load_model("./models/best_ensemble_model.pkl")


def prepare_input_basic(credit_score, location, gender, age, tenure, balance, num_products,
    has_credit_card, is_active_member, estimated_salary):
    """Prepare input for basic models (13 features)"""
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    
    expected_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany",
        "Geography_Spain", "Gender_Female", "Gender_Male"
    ]
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_columns]
    return input_df


def prepare_input_feature_engineered(credit_score, location, gender, age, tenure, balance, num_products,
    has_credit_card, is_active_member, estimated_salary):
    """Prepare input for feature engineered models (18 features)"""
    # Calculate engineered features
    customer_lifetime_value = (balance * estimated_salary) / 100000
    tenure_age_ratio = tenure / age if age > 0 else 0
    
    # Determine age group
    if age <= 30:
        age_group = "Young"
    elif age <= 45:
        age_group = "Middle-Aged"
    elif age <= 60:
        age_group = "Senior"
    else:
        age_group = "Elderly"
    
    # One-hot encode age group (Young is reference category, so excluded)
    age_group_middle_aged = 1 if age_group == "Middle-Aged" else 0
    age_group_senior = 1 if age_group == "Senior" else 0
    age_group_elderly = 1 if age_group == "Elderly" else 0

    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Customer Lifetime Value": customer_lifetime_value,
        "TenureAgeRatio": tenure_age_ratio,
        "AgeGroup_Middle-Aged": age_group_middle_aged,
        "AgeGroup_Senior": age_group_senior,
        "AgeGroup_Elderly": age_group_elderly,
    }

    expected_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany",
        "Geography_Spain", "Gender_Female", "Gender_Male", "Customer Lifetime Value",
        "TenureAgeRatio", "AgeGroup_Middle-Aged", "AgeGroup_Senior", "AgeGroup_Elderly"
    ]
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_columns]
    return input_df


def prepare_input_advanced(credit_score, location, gender, age, tenure, balance, num_products,
    has_credit_card, is_active_member, estimated_salary, balance_median):
    """Prepare input for advanced ensemble models (31 features)"""
    # Calculate basic engineered features
    customer_lifetime_value = (balance * estimated_salary) / 100000
    tenure_age_ratio = tenure / age if age > 0 else 0
    
    # Determine age group
    if age <= 30:
        age_group = "Young"
    elif age <= 45:
        age_group = "Middle-Aged"
    elif age <= 60:
        age_group = "Senior"
    else:
        age_group = "Elderly"
    
    # One-hot encode age group
    age_group_middle_aged = 1 if age_group == "Middle-Aged" else 0
    age_group_senior = 1 if age_group == "Senior" else 0
    age_group_elderly = 1 if age_group == "Elderly" else 0
    
    # Interaction features
    balance_per_product = balance / (num_products + 1)
    salary_to_balance_ratio = estimated_salary / (balance + 1) if balance > 0 else 0
    age_tenure_interaction = age * tenure
    credit_score_age_ratio = credit_score / age if age > 0 else 0
    
    # Risk indicator features (using median from training data)
    high_balance_low_activity = 1 if (balance > balance_median and is_active_member == 0) else 0
    low_products_high_balance = 1 if (num_products <= 1 and balance > balance_median) else 0
    new_customer_high_balance = 1 if (tenure <= 2 and balance > balance_median) else 0
    
    # Balance category (bins: [0, 1000, 50000, 100000, inf])
    if balance <= 1000:
        balance_category = "Low"
    elif balance <= 50000:
        balance_category = "Medium"
    elif balance <= 100000:
        balance_category = "High"
    else:
        balance_category = "VeryHigh"
    
    balance_category_medium = 1 if balance_category == "Medium" else 0
    balance_category_high = 1 if balance_category == "High" else 0
    balance_category_very_high = 1 if balance_category == "VeryHigh" else 0
    
    # Credit score category (bins: [0, 600, 700, 800, inf])
    if credit_score <= 600:
        credit_score_category = "Poor"
    elif credit_score <= 700:
        credit_score_category = "Fair"
    elif credit_score <= 800:
        credit_score_category = "Good"
    else:
        credit_score_category = "Excellent"
    
    credit_score_category_fair = 1 if credit_score_category == "Fair" else 0
    credit_score_category_good = 1 if credit_score_category == "Good" else 0
    credit_score_category_excellent = 1 if credit_score_category == "Excellent" else 0

    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Customer Lifetime Value": customer_lifetime_value,
        "TenureAgeRatio": tenure_age_ratio,
        "AgeGroup_Middle-Aged": age_group_middle_aged,
        "AgeGroup_Senior": age_group_senior,
        "AgeGroup_Elderly": age_group_elderly,
        "BalancePerProduct": balance_per_product,
        "SalaryToBalanceRatio": salary_to_balance_ratio,
        "AgeTenureInteraction": age_tenure_interaction,
        "CreditScoreAgeRatio": credit_score_age_ratio,
        "HighBalanceLowActivity": high_balance_low_activity,
        "LowProductsHighBalance": low_products_high_balance,
        "NewCustomerHighBalance": new_customer_high_balance,
        "BalanceCategory_Medium": balance_category_medium,
        "BalanceCategory_High": balance_category_high,
        "BalanceCategory_VeryHigh": balance_category_very_high,
        "CreditScoreCategory_Fair": credit_score_category_fair,
        "CreditScoreCategory_Good": credit_score_category_good,
        "CreditScoreCategory_Excellent": credit_score_category_excellent,
    }

    # Order matters for XGBoost models
    expected_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany",
        "Geography_Spain", "Gender_Female", "Gender_Male", "Customer Lifetime Value",
        "TenureAgeRatio", "AgeGroup_Middle-Aged", "AgeGroup_Senior", "AgeGroup_Elderly",
        "BalancePerProduct", "SalaryToBalanceRatio", "AgeTenureInteraction", "CreditScoreAgeRatio",
        "HighBalanceLowActivity", "LowProductsHighBalance", "NewCustomerHighBalance",
        "BalanceCategory_Medium", "BalanceCategory_High", "BalanceCategory_VeryHigh",
        "CreditScoreCategory_Fair", "CreditScoreCategory_Good", "CreditScoreCategory_Excellent"
    ]
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_columns]
    return input_df



def get_prediction_probability(model, input_df):
    """Get prediction probability from a model, handling hard voting classifiers"""
    try:
        # Try to get probability directly
        return model.predict_proba(input_df)[0][1]
    except AttributeError:
        # For hard voting classifiers, get probabilities from individual estimators
        try:
            # Check if it's a VotingClassifier with estimators
            if hasattr(model, 'estimators_'):
                # Get probabilities from each estimator and average them
                probas = []
                for name, estimator in model.named_estimators_.items():
                    try:
                        proba = estimator.predict_proba(input_df)[0][1]
                        probas.append(proba)
                    except (AttributeError, KeyError):
                        # If estimator doesn't support predict_proba, use predict
                        pred = estimator.predict(input_df)[0]
                        probas.append(float(pred))
                
                if probas:
                    return np.mean(probas)
            
            # Fallback: use predict and convert to probability
            prediction = model.predict(input_df)[0]
            return float(prediction)
        except Exception:
            # Final fallback: return 0.5 (uncertain)
            return 0.5


def make_predictions(credit_score, location, gender, age, tenure, balance, num_products,
    has_credit_card, is_active_member, estimated_salary, df):
    
    # Calculate balance median for risk indicator features
    balance_median = df['Balance'].median()
    
    # Prepare different feature sets for different model groups
    input_df_basic = prepare_input_basic(credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary)
    
    input_df_feature_engineered = prepare_input_feature_engineered(credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary)
    
    input_df_advanced = prepare_input_advanced(credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary, balance_median)

    probabilities = {
        "XGBoost": get_prediction_probability(xgb_model, input_df_basic),
        "Naive Bayes": get_prediction_probability(naive_bayes_model, input_df_basic),
        "Random Forest": get_prediction_probability(rf_model, input_df_basic),
        "Decision Tree": get_prediction_probability(decision_tree_model, input_df_basic),
        "KNN": get_prediction_probability(knn_model, input_df_basic),

        "XGBoost SMOTE": get_prediction_probability(xgboost_SMOTE_model, input_df_feature_engineered),
        "XGBoost Feature Engineered": get_prediction_probability(xgboost_featureEngineered_model, input_df_feature_engineered),
        "Voting Classifier": get_prediction_probability(voting_classifier_model, input_df_feature_engineered),

        "Soft Voting (⭐️)": get_prediction_probability(soft_voting_model, input_df_advanced),
        "Weighted Voting (⭐️)": get_prediction_probability(weighted_voting_model, input_df_advanced),
        "Best Ensemble (⭐️)": get_prediction_probability(best_ensemble_model, input_df_advanced),
    }

    # Calculate the average probability of the best three models
    last_three_probs = [
        probabilities["Soft Voting (⭐️)"],
        probabilities["Weighted Voting (⭐️)"],
        probabilities["Best Ensemble (⭐️)"]
    ]

    avg_probability = np.mean(last_three_probs)

    with st.expander("Model Probabilities"):
        for model, prob in probabilities.items():
            st.write(f"{model}: {prob * 100:.2f}%")
        
        st.write(f"Average Probability based on the best three models (⭐️): {avg_probability * 100:.2f}%")



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

        location_options = ["France", "Germany", "Spain"]
        location = st.selectbox("Location", options=location_options, index=location_options.index(selected_customer['Geography']))

        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == "Male" else 1)

        age = st.number_input("Age", value=int(selected_customer['Age']), min_value=18, max_value=100, step=1)

        tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, step=1, value=int(selected_customer['Tenure']))
    
    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer['Balance']))

        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts']))

        has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard']))

        is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer['EstimatedSalary']))

        
    make_predictions(credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary, df)



    

