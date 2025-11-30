import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML models API for predicting customer churn",
    version="1.0.0"
)

# Path configuration - adjust based on where the script is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = BASE_DIR  # Models are in the same directory as deploy.py
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")


def load_model(filename):
    """Load a pickled model from file"""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


# Load all models at startup
print("Loading models...")
try:
    # OG models
    xgb_model = load_model("xgb_model.pkl")
    naive_bayes_model = load_model("nb_model.pkl")
    rf_model = load_model("rf_model.pkl")
    svm_model = load_model("svm_model.pkl")
    decision_tree_model = load_model("dt_model.pkl")
    knn_model = load_model("knn_model.pkl")

    # Middle models
    xgboost_SMOTE_model = load_model("xgbboost-SMOTE.pkl")
    xgboost_featureEngineered_model = load_model("xgbboost-featureEngineered.pkl")
    voting_classifier_model = load_model("voting_clf.pkl")

    # Ensemble models
    soft_voting_model = load_model("voting_clf.pkl")
    weighted_voting_model = load_model("weighted_voting_tuned.pkl")
    best_ensemble_model = load_model("best_ensemble_model.pkl")
    
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Load data for balance median calculation
try:
    data_path = os.path.join(DATA_DIR, "churn.csv")
    df = pd.read_csv(data_path)
    BALANCE_MEDIAN = df['Balance'].median()
    print(f"Data loaded. Balance median: {BALANCE_MEDIAN}")
except Exception as e:
    print(f"Warning: Could not load data for balance median. Using default value. Error: {e}")
    BALANCE_MEDIAN = 0.0  # Fallback value


# Pydantic models for request/response
class CustomerInput(BaseModel):
    credit_score: int = Field(..., ge=350, le=850, description="Credit score (350-850)")
    location: Literal["France", "Germany", "Spain"] = Field(..., description="Customer location")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    tenure: int = Field(..., ge=0, le=50, description="Tenure in years (0-50)")
    balance: float = Field(..., ge=0, description="Account balance")
    num_products: int = Field(..., ge=1, le=10, description="Number of products (1-10)")
    has_credit_card: bool = Field(..., description="Has credit card")
    is_active_member: bool = Field(..., description="Is active member")
    estimated_salary: float = Field(..., ge=0, description="Estimated salary")

    class Config:
        json_schema_extra = {
            "example": {
                "credit_score": 619,
                "location": "France",
                "gender": "Female",
                "age": 42,
                "tenure": 2,
                "balance": 0.0,
                "num_products": 1,
                "has_credit_card": True,
                "is_active_member": True,
                "estimated_salary": 101348.88
            }
        }


class ModelPrediction(BaseModel):
    model_name: str
    probability: float
    percentage: float


class PredictionResponse(BaseModel):
    average_probability: float
    average_percentage: float
    individual_predictions: list[ModelPrediction]
    best_three_models: list[ModelPrediction]


# Input preparation functions (from app/main.py)
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


# API Endpoints
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "balance_median": float(BALANCE_MEDIAN)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    """
    Predict customer churn probability using all available ML models.
    
    Returns predictions from all models and the average probability from the best three ensemble models.
    """
    try:
        # Prepare different feature sets for different model groups
        input_df_basic = prepare_input_basic(
            customer.credit_score, customer.location, customer.gender, customer.age, 
            customer.tenure, customer.balance, customer.num_products, 
            customer.has_credit_card, customer.is_active_member, customer.estimated_salary
        )
        
        input_df_feature_engineered = prepare_input_feature_engineered(
            customer.credit_score, customer.location, customer.gender, customer.age, 
            customer.tenure, customer.balance, customer.num_products, 
            customer.has_credit_card, customer.is_active_member, customer.estimated_salary
        )
        
        input_df_advanced = prepare_input_advanced(
            customer.credit_score, customer.location, customer.gender, customer.age, 
            customer.tenure, customer.balance, customer.num_products, 
            customer.has_credit_card, customer.is_active_member, customer.estimated_salary, 
            BALANCE_MEDIAN
        )

        # Get predictions from all models
        probabilities = {
            "XGBoost": get_prediction_probability(xgb_model, input_df_basic),
            "Naive Bayes": get_prediction_probability(naive_bayes_model, input_df_basic),
            "Random Forest": get_prediction_probability(rf_model, input_df_basic),
            "Decision Tree": get_prediction_probability(decision_tree_model, input_df_basic),
            "KNN": get_prediction_probability(knn_model, input_df_basic),
            "XGBoost SMOTE": get_prediction_probability(xgboost_SMOTE_model, input_df_feature_engineered),
            "XGBoost Feature Engineered": get_prediction_probability(xgboost_featureEngineered_model, input_df_feature_engineered),
            "Voting Classifier": get_prediction_probability(voting_classifier_model, input_df_feature_engineered),
            "Soft Voting": get_prediction_probability(soft_voting_model, input_df_advanced),
            "Weighted Voting": get_prediction_probability(weighted_voting_model, input_df_advanced),
            "Best Ensemble": get_prediction_probability(best_ensemble_model, input_df_advanced),
        }

        # Calculate the average probability of the best three models
        best_three_probs = [
            probabilities["Soft Voting"],
            probabilities["Weighted Voting"],
            probabilities["Best Ensemble"]
        ]
        avg_probability = np.mean(best_three_probs)

        # Format response
        individual_predictions = [
            ModelPrediction(
                model_name=name,
                probability=float(prob),
                percentage=float(prob * 100)
            )
            for name, prob in probabilities.items()
        ]

        best_three_models = [
            ModelPrediction(
                model_name="Soft Voting",
                probability=float(probabilities["Soft Voting"]),
                percentage=float(probabilities["Soft Voting"] * 100)
            ),
            ModelPrediction(
                model_name="Weighted Voting",
                probability=float(probabilities["Weighted Voting"]),
                percentage=float(probabilities["Weighted Voting"] * 100)
            ),
            ModelPrediction(
                model_name="Best Ensemble",
                probability=float(probabilities["Best Ensemble"]),
                percentage=float(probabilities["Best Ensemble"] * 100)
            )
        ]

        return PredictionResponse(
            average_probability=float(avg_probability),
            average_percentage=float(avg_probability * 100),
            individual_predictions=individual_predictions,
            best_three_models=best_three_models
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
