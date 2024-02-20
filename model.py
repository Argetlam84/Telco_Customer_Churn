
import numpy as np
import pandas as pd
import feature_engineering as fe
import variable_evaluations as ve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression




def base_model_results(df, cat_cols, target_col="Churn", drop_first=True, random_state=12345):
    dff = df.copy()
    dff = pd.get_dummies(dff, columns=[col for col in cat_cols if col != target_col], drop_first=drop_first)

    y = dff[target_col]
    X = dff.drop([target_col, "customerID"], axis=1)


    models = [('LR', LogisticRegression(random_state=random_state)),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=random_state)),
              ('RF', RandomForestClassifier(random_state=random_state)),
              ('XGB', XGBClassifier(random_state=random_state)),
              ("LightGBM", LGBMClassifier(random_state=random_state)),
              ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]


    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"AUC: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
        print("\n")




def train_catboost(X, y, params, cv=5):
    catboost_model = CatBoostClassifier(random_state=17, verbose=False)
    catboost_grid = GridSearchCV(catboost_model, params, cv=cv, n_jobs=-1, verbose=True)
    catboost_grid.fit(X, y)
    best_params = catboost_grid.best_params_
    catboost_final = catboost_model.set_params(**best_params, random_state=17)
    catboost_final.fit(X, y)
    scoring = ["accuracy", "precision", "recall"]
    cv_results = cross_validate(catboost_final, X, y, cv=cv, scoring=scoring, return_train_score=True)
    return {
        "best_params": best_params,
        "model": catboost_final,
        "accuracy": cv_results['test_accuracy'].mean(),
        "precision": cv_results['test_precision'].mean(),
        "recall": cv_results['test_recall'].mean(),
    }


def train_logistic_regression(X, y, logistic_params, cv=5):
    logistic_model = LogisticRegression()

    grid_search = GridSearchCV(estimator=logistic_model, param_grid=logistic_params, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    logistic_final = logistic_model.set_params(**best_params)

    scoring = ["accuracy", "precision", "recall"]
    cv_results = cross_validate(logistic_final, X, y, cv=cv, scoring=scoring, return_train_score=True)

    accuracy = cv_results['test_accuracy'].mean()
    precision = cv_results['test_precision'].mean()
    recall = cv_results['test_recall'].mean()

    return {
        "best_params": best_params,
        "model": logistic_final,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }



def evaluate_classification_models(X, y, plot_imp=False, save=False, num=20, random_state=None):

    global fitted_models

    models = [
        ('XGB', XGBClassifier(random_state=random_state)),
        ("Logistic Regression", LogisticRegression(random_state=random_state)),
        ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state)),
        ('LR', LogisticRegression(random_state=random_state)),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(random_state=random_state)),
        ('RF', RandomForestClassifier(random_state=random_state)),
        ("LightGBM", LGBMClassifier(random_state=random_state))
    ]

    models_names = {}
    performance = {}

    for model_name, model in models:
        performance[model_name] = {
            "Accuracy": np.mean(cross_val_score(model, X, y, cv=5, scoring="accuracy")),
            "AUC": np.mean(cross_val_score(model, X, y, cv=5, scoring="roc_auc")),
            "Recall": np.mean(cross_val_score(model, X, y, cv=5, scoring="recall")),
            "Precision": np.mean(cross_val_score(model, X, y, cv=5, scoring="precision")),
            "F1 Score": np.mean(cross_val_score(model, X, y, cv=5, scoring="f1"))
        }
        models_names[model_name] = model

        fitted_models = [model.fit(X, y) for model_name, model in models]

    if plot_imp:
        ve.plot_importance_for_func(fitted_models, X, num=num, save=save)

    return performance, models_names

def find_top_models(X, y, num_top_models=2, random_state=None):
    models = [
        ('XGB', XGBClassifier(random_state=random_state)),
        ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state)),
        ('LR', LogisticRegression(random_state=random_state)),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(random_state=random_state)),
        ('RF', RandomForestClassifier(random_state=random_state)),
        ("LightGBM", LGBMClassifier(random_state=random_state))
    ]

    performance = {}

    for model_name, model in models:
        performance[model_name] = {
            "Accuracy": np.mean(cross_val_score(model, X, y, cv=5, scoring="accuracy")),
            "Precision": np.mean(cross_val_score(model, X, y, cv=5, scoring="precision")),
            "Recall": np.mean(cross_val_score(model, X, y, cv=5, scoring="recall"))
        }

    sorted_models = sorted(performance.items(), key=lambda x: (x[1]["Accuracy"], x[1]["Precision"], x[1]["Recall"]), reverse=True)

    top_models = sorted_models[:num_top_models]

    return top_models