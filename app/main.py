import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)


def explain_prediction(probability, input_dict, surname):
    prompt = f"""
        You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

        Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)} percent probability of churning, based on the information provided below.

        Here is the customer's information:
        {input_dict}

        Here are the machine learning model's top 10 most important features for predicting churn:

        Feature | Importance
        --------------------
        NumOfProducts     | 0.323888
        IsActiveMember    | 0.164146
        Age               | 0.109550
        Geography_Germany | 0.091373
        Balance           | 0.052786
        Geography_France  | 0.046463
        Gender_Female     | 0.045283
        Geography_Spain   | 0.036855
        CreditScore       | 0.036005
        EstimatedSalary   | 0.032655
        HasCrCard         | 0.031940
        Tenure            | 0.030054
        Gender_Male       | 0.000000


        {pd.set_option('display.max_columns', None)}

        Here are summary statistics for churned customers:
        {df[df['Exited'] == 1].describe()}

        Here are summary statistics for non-churned customers:
        {df[df['Exited'] == 0].describe()}

        - If the customer has OVER a 40 percent risk of churning, generate a 3-5 sentence explanation of why they are at risk of churning.
        - If the customer has LESS THAN a 40 percent risk of churning, generate a 3-5 sentence explanation of why they might not be at risk of churning.
        - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

        Do not:
        - Mention the probability of churning.
        - Mention the actual percentage probability of churning as it will already be shown on the dashboard.
        - Mention machine learning, models, predictions, algorithms, AI, training data, or feature importances.
        - Use phrases like "the model thinks", "based on the prediction", "the system shows", or anything that reveals that an automated model is involved.

        You are talking to an internal bank analyst, so you should be professional and use business language as they may not be familiar with the technical 
        details of the model or the dataset.
       """

    print("EXPLANATION PROMPT:", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    # Convert probability to percentage for internal docs
    churn_pct = round(probability * 100, 1)

    # Define incentive bundles by churn tier
    if probability < 0.30:
        risk_tier = "low"
        incentive_bundle = """
            Low-risk incentive bundle (internal only):
            - Waive one month of maintenance or service fees.
            - Offer a small welcome-back bonus of loyalty points or cash (for example, $25–$50) if they keep their primary checking account active.
            - Provide a small rate boost on a savings account for the next 6–12 months.
            - Offer a quick financial check-in call to optimize their current products, without any pressure to add new ones.
        """
    elif probability < 0.60:
        risk_tier = "medium"
        incentive_bundle = """
            Medium-risk incentive bundle (internal only):
            - Waive 3 months of maintenance or service fees on their primary account.
            - Offer a more meaningful cash-back or statement-credit bonus (for example, $75–$150) if they keep their accounts active and meet simple usage criteria.
            - Provide a preferential rate on a savings, CD, or money market account for 12 months.
            - Invite them to a personalized review with a dedicated banker to consolidate accounts, adjust limits, and tailor products to their goals.
        """
    else:
        risk_tier = "high"
        incentive_bundle = """
            High-risk incentive bundle (internal only):
            - Waive overdraft, maintenance, and transfer fees for the next 3–6 months, where policy allows.
            - Offer a substantial retention bonus (for example, $150–$300) in loyalty points or statement credit if they keep their primary relationship and meet simple activity criteria.
            - Provide a top-tier promotional rate on savings/CDs or a reduced rate on eligible loans, subject to credit approval.
            - Assign a dedicated relationship manager to them, with priority access by phone or email.
            - Offer to simplify and consolidate multiple accounts, automate key payments, and set up alerts tailored to their habits.
        """

    prompt = f"""
        You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with thoughtful, customer-friendly offers.

        INTERNAL INFORMATION (DO NOT MENTION DIRECTLY IN THE EMAIL):
        - Customer surname: {surname}
        - Internal churn risk percentage: {churn_pct}%
        - Internal churn risk tier: {risk_tier.upper()}
        - Why we believe they might consider leaving:
        {explanation}
        - Incentive bundle to use for this customer:
        {incentive_bundle}

        CUSTOMER INFORMATION (YOU MAY USE THIS IN THE EMAIL):
        {input_dict}

        WRITING TASK:
        Write a warm, engaging, and professional email to this customer that encourages them to stay with the bank.

        Email requirements:
        - Include a short, friendly subject line.
        - Start with a personal greeting using their last name (e.g., "Dear Mr. {surname}" or "Dear {surname} family" if last name is ambiguous).
        - In the opening paragraph, thank them for being a customer and acknowledge the value of their relationship with the bank.
        - In the next 1–2 paragraphs, briefly highlight how the bank can support their goals based on the customer information.
        - Then include a section such as "Here are a few ways we can support you:" followed by bullet points.
        - The bullet points must be a clear, customer-friendly rephrasing of the incentives from the incentive bundle above that corresponds to their risk tier. Do NOT invent new types of incentives beyond what is there, but you may adjust the exact wording to sound natural.
        - Close with a clear, low-pressure call to action (for example, inviting them to reply to the email, schedule a call, or visit a branch) and a warm sign-off.

        STYLE CONSTRAINTS:
        - Do NOT mention churn risk, probabilities, models, AI, algorithms, predictions, or anything about internal scoring.
        - Do NOT mention "risk tier" or "incentive bundle"; that is internal language only.
        - The email should sound like it was written by a human relationship manager who genuinely wants to help the customer, not by an automated system.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    print("\n\nEMAIL PROMPT\n", prompt)

    return raw_response.choices[0].message.content



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

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability * 100:.2f}% risk of churning.")
    
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)


    return avg_probability


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

    
    input_dict = prepare_input_basic(credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary)


    avg_probability = make_predictions(credit_score, location, gender, age, tenure, balance, 
        num_products, has_credit_card, is_active_member, estimated_salary, df)

    # Calculate and display customer percentiles
    st.markdown("------")
    st.subheader("Customer Percentiles")
    
    customer_values = {
        'CreditScore': credit_score,
        'Balance': balance,
        'EstimatedSalary': estimated_salary,
        'Tenure': tenure,
        'NumOfProducts': num_products
    }
    
    percentiles = ut.calculate_customer_percentiles(customer_values, df)
    fig_percentiles = ut.create_customer_percentiles_chart(percentiles)
    st.plotly_chart(fig_percentiles, use_container_width=True)

    explanation = explain_prediction(avg_probability, input_dict, selected_surname)

    st.markdown("------")
    st.subheader("Explanation of Prediction")

    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation, selected_surname)

    st.markdown("------")

    st.subheader("Personalized Retention Email")

    st.markdown(email)
    

