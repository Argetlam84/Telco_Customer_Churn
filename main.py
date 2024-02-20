

import data_reading_and_understanding as dr
import feature_engineering as fe
import variable_evaluations as ve
import model as ml
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("Datasets/Telco-Customer-Churn.csv")

#EDA
dr.check_data(df)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

cat_cols, num_cols, cat_but_car = dr.grab_col_names(df)


for col in cat_cols:
    dr.cat_summary(df, col, plot=False)
"""Approximately half of our dataset's customers are male, and the other half are female.
About 50% of the customers have a partner (married etc...).
Only 30% of the total customers have dependents to care for.
90% of the customers receive telephone service.
Among those receiving telephone service, 53% do not have multiple lines.
There is a 21% segment of the customer base that does not have an internet service provider.
Most customers opt for month-to-month contracts. There are similar numbers of customers on 1-year and 2-year contracts.
60% of the customers have paperless billing.
Approximately 26% of the customers churned from the platform .
16% of the dataset consists of elderly customers. Hence, the majority of the customers in the dataset are young.
"""

for col in num_cols:
    dr.num_summary(df, col, plot=True)
"""
When we look at tenure, we see that there is a large number of customers with 1-month tenure, 
followed by customers with 70 months of tenure.
"""

for col in num_cols:
    dr.target_summary_with_num(df, "Churn", col)
"""
When we examine the relationship between tenure and churn, we see that customers who do not churn have been customers for a longer period.

When we look at monthly charges and churn, we find that the average monthly charges of churned customers are higher.
"""

for col in cat_cols:
    dr.target_summary_with_cat(df, "Churn", col)
"""
Churn percentages are nearly equal between males and females.
Customers with partners and dependents tend to have lower churn rates.
There is no significant difference in churn rates for customers with PhoneService and MultipleLines.
Loss rates are significantly higher for Fiber Optic Internet Services.
Customers without services like OnlineSecurity, OnlineBackup, and TechSupport have higher churn rates.
A greater percentage of customers with monthly subscriptions churn compared to those with one or two-year contracts.
Customers with paperless billing experience higher churn rates.
Customers with ElectronicCheck as their PaymentMethod tend to churn more compared to other options.
Churn percentages are higher among elderly customers.
"""

dr.correlation_analysis(dataframe=df, num_cols=["TotalCharges", "MonthlyCharges", "tenure"], target_col="Churn")



#FEATURE ENGINEERING
na_columns = fe.missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["tenure"].min()
df["tenure"] = df["tenure"] + 1


for col in num_cols:
    print(col, fe.check_outlier(df, col))


#BASE MODEL
ml.base_model_results(df, cat_cols)
"""########## LR ##########
Accuracy: 0.8035
AUC: 0.8425
Recall: 0.5388
Precision: 0.6589
F1: 0.5924
########## KNN ##########
Accuracy: 0.7629
AUC: 0.7461
Recall: 0.4468
Precision: 0.5686
F1: 0.4999
########## CART ##########
Accuracy: 0.728
AUC: 0.6586
Recall: 0.5077
Precision: 0.4886
F1: 0.4977
########## RF ##########
Accuracy: 0.792
AUC: 0.8252
Recall: 0.4842
Precision: 0.6448
F1: 0.5529
########## XGB ##########
Accuracy: 0.7833
AUC: 0.8228
Recall: 0.5072
Precision: 0.6123
F1: 0.5542
########## LightGBM ##########
Accuracy: 0.7982
AUC: 0.8373
Recall: 0.5281
Precision: 0.6482
F1: 0.5816
########## CatBoost ##########
Accuracy: 0.797
AUC: 0.8401
Recall: 0.5051
Precision: 0.6531
F1: 0.5691
"""
#CREATING NEW FEATURES
df = fe.new_feature_eng_cols(df)

#ENCODING
cat_cols, num_cols, cat_but_car = dr.grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
for col in binary_cols:
    df = fe.label_encoder(df, col)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
df = fe.one_hot_encoder(df, cat_cols, drop_first=True)


#MODELLING
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)


#MODELS' PERFORMANCES
performance, models = ml.evaluate_classification_models(X, y, plot_imp=True)

#TO FIND TOP 2 MODELS
top_models = ml.find_top_models(X, y)
for model_name, performance_metrics in top_models:
    print("Model:", model_name)
    print("Performance:", performance_metrics)

#HYPERPARAMETER OPTIMIZATION

#CATBOOST
catboost_params = {
    "iterations": [600, 700],
    "learning_rate": [0.015, 0.02],
    "depth": [4, 10]
}


catboost_results = ml.train_catboost(X, y, catboost_params)

print("CatBoost Best Parameters:", catboost_results["best_params"])
print("CatBoost Accuracy: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(catboost_results["accuracy"], catboost_results["recall"], catboost_results["precision"]))


# LOGISTIC REGRESSION (THE BEST ONE)

logistic_params = {
    "C": [10, 12, 15],
    "solver": ['liblinear', 'newton-cg'],
    "max_iter": [90, 100, 110]
}

logistic_results = ml.train_logistic_regression(X, y, logistic_params)
print("Logistic Regression Best Parameters:", logistic_results["best_params"])
print("Logistic Regression Accuracy: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(logistic_results["accuracy"], logistic_results["recall"], logistic_results["precision"]))








