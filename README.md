# 💳 Credit Risk Prediction
**Leveraging Machine Learning for Credit Card Default Detection**  
_Group Project for Machine Learning II (IE University)_


## 👥 Authors

Ana Cortés Barquier, Tomás Valbuena Sierra, Tomás Luz, Robert Koegel, Hiromitsu Fujiyama


## 🧠 Project Overview

This project applies multiple machine learning algorithms to **predict the probability of customer default** on credit card payments. By using demographic and financial variables, we built, tuned, and compared classification models to guide credit risk decisions in financial institutions.

> 📌 **Business Goal**: Maximize Recall to identify as many potential defaulters as possible, even at the cost of more false positives. This conservative approach helps banks reduce missed defaults and minimize financial losses.


## 📊 Dataset Summary

- **Source**: Provided for academic use
- **Size**: ~30,000 customer records
- **Target**: `default_payment_next_month` (1 = default, 0 = no default)
- **Class Imbalance**: 22% defaulters, 78% non-defaulters

### 🔑 Key Features:
- 'LIMIT_BAL' – credit limit
- 'PAY_AMT' – recent payment amounts
- 'BILL_AMT' – outstanding bills
- 'PAYMENT_DELAY', 'RC' – payment behaviors
- Demographics: age, sex, education, marital status

> ⚠️ **Note**: The dataset cannot be shared due to academic constraints.


## 🧪 Models and Methodology

We tested the following classification algorithms:

| Model            | Recall (↑) | AUC (↑) | Precision (↑) | Notes |
|------------------|------------|---------|----------------|-------|
| **Bagging**      | **56.5%**  | **70.3%** | 49.8%        | Best model for default detection |
| Random Forest    | 55.6%      | 70.2%   | 50.5%          | Strong overall performance |
| XGBoost          | 55.6%      | 67.6%   | 47.2%          | Lower AUC than RF |
| Decision Tree    | 46.4%      | 68.3%   | 55.9%          | Simple and interpretable |
| KNN              | 35.1%      | 66.8%   | **66.1%**      | Highest precision but misses defaulters |

> Threshold tuned to **0.3** to improve Recall while balancing AUC and Precision.


## ⚙️ Model Development Highlights

- Feature engineering for **payment delay patterns** and **revolving credit**
- Stratified train-test split (50/50) to balance default class
- Hyperparameter tuning via **GridSearchCV** and **RandomizedSearchCV**
- Evaluation metrics: **Recall, AUC, Precision, Accuracy**


## 📈 Feature Importance

Top predictors (via information gain):
- Payment delays ('PAYMENT_MONTH_DELAY')
- Recent payments ('PAY_AMT')
- Credit limits ('LIMIT_BAL')
- Outstanding bills ('BILL_AMT')

> Demographic variables had minimal predictive power.


## 🧩 Recommendations for Risk Management

- ✅ Use **Bagging** to flag high-risk accounts for early intervention
- 📉 Lower credit limits for high-risk users; increase for low-risk users
- 🔄 Integrate model into real-time decision-making for loans
- 🎯 Use predictions to guide marketing, collections, and customer support


## 📂 Project Structure
- README.md >>> Project overview, methods, and results (this file)
- Credit_Dataset.csv >>> Dataset containing financial and demographic data for credit default prediction
- Credit_Prediction_Notebook.ipynb >>> Jupyter Notebook with full pipeline: preprocessing, modeling, evaluation


## 🛠 Tools & Libraries

- Python (NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- GridSearchCV, RandomizedSearchCV, StandardScaler
