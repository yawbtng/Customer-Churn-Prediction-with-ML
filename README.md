# Customer Churn Prediction with Machine Learning

**Course**: CS 5393 - Introduction to Machine Learning  
**Student**: Yaw Boateng  
**Date**: October 11th, 2025

---

## Project Overview

This project aims to develop a comprehensive machine learning pipeline for predicting customer churn in the banking sector. Using a dataset of 10,000 bank customers, we will build and compare multiple regression models to predict churn probability, ultimately deploying the best-performing model through a web application for real-time predictions.

## Dataset Information

### Source and Description
- **Dataset**: Bank Customer Churn Dataset
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
- **Size**: 10,000 rows (excluding header)
- **Features**: 14 columns total

### Dataset Structure

**Independent Variables (13 features)**:
- `RowNumber`: Sequential row identifier
- `CustomerId`: Unique customer identifier
- `Surname`: Customer's last name
- `CreditScore`: Customer's credit score (300-850)
- `Geography`: Country of residence (France, Germany, Spain)
- `Gender`: Customer gender (Male, Female)
- `Age`: Customer age
- `Tenure`: Number of years as bank customer
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products used
- `HasCrCard`: Credit card ownership (0/1)
- `IsActiveMember`: Active membership status (0/1)
- `EstimatedSalary`: Estimated annual salary

**Target Variable (1 feature)**:
- `Exited`: Churn indicator (0 = stayed, 1 = churned)

### Data Types and Characteristics
- **Numerical Features**: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary
- **Categorical Features**: Geography, Gender, HasCrCard, IsActiveMember
- **Binary Features**: HasCrCard, IsActiveMember, Exited

## Machine Learning Task Definition

### Task Type: Regression

**Objective**: Predict customer churn probability as a continuous value between 0 and 1.

### Problem Formulation
- **Input**: Customer features (CreditScore, Geography, Gender, Age, Tenure, Balance, etc.)
- **Output**: Churn probability (continuous value from 0 to 1)
- **Goal**: Minimize prediction error for churn probability estimation

### Justification for Regression Approach
While customer churn is inherently binary (churned vs. stayed), we choose regression for several reasons:

1. **Probability Estimation**: Regression allows us to predict the probability of churn, providing more nuanced insights than binary classification
2. **Business Value**: Banks can prioritize customers based on churn probability scores for targeted retention campaigns
3. **Risk Assessment**: Continuous probability scores enable better risk stratification and resource allocation
4. **Model Flexibility**: Regression models can be easily thresholded for binary decisions while maintaining probabilistic interpretation

### Success Metrics
- **Primary**: Root Mean Square Error (RMSE)
- **Secondary**: Mean Absolute Error (MAE), R² Score
- **Business**: Ability to identify high-risk customers for retention efforts

## Project Motivation

### Learning Objectives
This project serves multiple educational and practical purposes:

1. **Practical ML Application**: Apply machine learning techniques to a real-world business problem that affects millions of customers globally
2. **End-to-End Pipeline Development**: Learn to build complete ML workflows from data preprocessing to model deployment
3. **Algorithm Comparison**: Gain hands-on experience comparing different machine learning algorithms and understanding their strengths/weaknesses
4. **Business Analytics Integration**: Understand how ML models can drive business decisions and create value

### Business Relevance
Customer churn prediction is critical for the banking industry:

- **Financial Impact**: Customer acquisition costs are 5-25x higher than retention costs
- **Revenue Protection**: Preventing churn directly protects revenue streams
- **Competitive Advantage**: Proactive retention strategies improve customer satisfaction and loyalty
- **Resource Optimization**: Targeted interventions are more cost-effective than blanket retention programs

## Proposed Approach

### 1. Data Preprocessing

**Data Cleaning**:
- Handle missing values using appropriate imputation strategies
- Detect and treat outliers using IQR method or domain knowledge
- Remove irrelevant features (RowNumber, CustomerId, Surname)

**Feature Engineering**:
- One-hot encode categorical variables (Geography, Gender)
- Create interaction features (Age × Balance, Tenure × NumOfProducts)
- Normalize numerical features using StandardScaler
- Handle class imbalance if present

**Data Splitting**:
- Train/Validation/Test split: 70%/15%/15%
- Stratified sampling to maintain churn distribution
- Cross-validation for robust model evaluation

### 2. Model Selection (5 Models)

We will implement and compare the following regression models:

1. **Linear Regression**
   - Baseline model for comparison
   - Fast training and interpretable coefficients
   - Assumes linear relationship between features and target

2. **K-Nearest Neighbors (KNN)**
   - Non-parametric method
   - Good for capturing local patterns
   - Sensitive to feature scaling

3. **Support Vector Regression (SVR)**
   - Effective for non-linear relationships
   - Robust to outliers
   - Kernel trick for complex feature interactions

4. **Random Forest Regressor**
   - Ensemble method combining multiple decision trees
   - Handles non-linear relationships and feature interactions
   - Provides feature importance rankings

5. **XGBoost Regressor**
   - Gradient boosting ensemble method
   - Often achieves state-of-the-art performance
   - Built-in regularization and feature importance

### 3. Hyperparameter Tuning

**Optimization Strategy**:
- Use GridSearchCV for smaller parameter spaces
- Use RandomizedSearchCV for larger parameter spaces
- 5-fold cross-validation for robust evaluation
- Optimize for RMSE as primary metric

**Key Hyperparameters**:
- **KNN**: n_neighbors, weights, distance metric
- **SVR**: C, gamma, kernel type
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: learning_rate, max_depth, n_estimators, subsample

### 4. Model Evaluation

**Performance Metrics**:
- **RMSE**: Primary metric for regression evaluation
- **MAE**: Robust to outliers, interpretable
- **R² Score**: Proportion of variance explained
- **Cross-validation**: 5-fold CV for robust performance estimates

**Model Comparison**:
- Statistical significance testing between models
- Learning curves to assess bias-variance tradeoff
- Feature importance analysis for business insights
- Residual analysis for model diagnostics

### 5. Deployment Strategy

**Web Application Development**:
- Build interactive web interface using Flask/Streamlit
- User-friendly input forms for customer data
- Real-time churn probability predictions
- Visualization of prediction confidence and feature contributions

**Model Serving**:
- Save trained models using joblib/pickle
- Implement model versioning and A/B testing capabilities
- Add input validation and error handling
- Create API endpoints for integration with other systems

## Expected Outcomes

### Technical Deliverables

1. **Trained Models**
   - Five optimized regression models with hyperparameter tuning
   - Performance comparison across all models
   - Best-performing model selected based on RMSE and business criteria

2. **Model Performance Analysis**
   - Comprehensive evaluation metrics (RMSE, MAE, R²)
   - Cross-validation results for robust performance estimates
   - Feature importance rankings and business insights
   - Statistical significance testing between models

3. **Data Insights**
   - Key factors driving customer churn
   - Customer segmentation based on churn risk
   - Recommendations for retention strategies
   - Visualization of model performance and feature contributions

4. **Web Application**
   - Interactive interface for churn probability prediction
   - Real-time model inference capabilities
   - User-friendly input forms and result visualization
   - API endpoints for integration with other systems

### Business Value

1. **Customer Retention Strategy**
   - Identify high-risk customers for proactive intervention
   - Prioritize retention efforts based on churn probability
   - Optimize resource allocation for customer service

2. **Risk Management**
   - Early warning system for customer churn
   - Data-driven approach to customer lifecycle management
   - Improved customer satisfaction through targeted interventions

3. **Operational Efficiency**
   - Automated churn prediction reduces manual analysis
   - Scalable solution for large customer bases
   - Integration capabilities with existing banking systems

## Project Timeline and Milestones

### Phase 1: Data Exploration and Preprocessing (Week 1)
- [ ] Load and explore the dataset
- [ ] Perform exploratory data analysis (EDA)
- [ ] Handle missing values and outliers
- [ ] Engineer features and encode categorical variables
- [ ] Split data into train/validation/test sets

### Phase 2: Model Development (Week 2)
- [ ] Implement Linear Regression baseline model
- [ ] Implement K-Nearest Neighbors model
- [ ] Implement Support Vector Regression model
- [ ] Implement Random Forest Regressor
- [ ] Implement XGBoost Regressor

### Phase 3: Model Optimization (Week 3)
- [ ] Perform hyperparameter tuning for all models
- [ ] Cross-validation and performance evaluation
- [ ] Model comparison and selection
- [ ] Feature importance analysis
- [ ] Statistical significance testing

### Phase 4: Deployment and Documentation (Week 4)
- [ ] Build web application for model inference
- [ ] Create user interface and API endpoints
- [ ] Test and validate deployed model
- [ ] Document results and create visualizations
- [ ] Prepare final presentation and report


## Conclusion

This project will provide comprehensive experience in applying machine learning to solve real-world business problems. By developing a complete pipeline from data preprocessing to model deployment, we will gain valuable insights into customer churn prediction while building practical skills in machine learning, data science, and web development.

The combination of technical rigor and business relevance makes this project an excellent learning opportunity that bridges academic concepts with practical applications in the banking industry.

