import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def base_model_results(df, cat_cols, target_col="Churn", drop_first=True, random_state=12345):
    # Create a copy of the DataFrame
    dff = df.copy()

    # Perform one-hot encoding on categorical columns
    dff = pd.get_dummies(dff, columns=[col for col in cat_cols if col != target_col], drop_first=drop_first)

    # Separate features and target variable
    y = dff[target_col]
    X = dff.drop([target_col, "customerID"], axis=1)

    # Define models
    models = [('LR', LogisticRegression(random_state=random_state)),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=random_state)),
              ('RF', RandomForestClassifier(random_state=random_state)),
              ('XGB', XGBClassifier(random_state=random_state)),
              ("LightGBM", LGBMClassifier(random_state=random_state)),
              ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

    # Iterate over models and perform cross-validation
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"AUC: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
        print("\n")

def new_feature_eng_cols(dataframe):
    """
    Generate new features from existing columns in the DataFrame.

    Parameters:
    dataframe (DataFrame): Input DataFrame containing customer data.

    Returns:
    DataFrame: DataFrame with new features added.
    """
    # Create a new feature 'NEW_TENURE' based on tenure duration
    dataframe.loc[dataframe["tenure"] <= 12, "NEW_TENURE"] = "0-1 Year"
    dataframe.loc[(dataframe["tenure"] > 12) & (dataframe["tenure"] <= 24), "NEW_TENURE"] = "1-2 Year"
    dataframe.loc[(dataframe["tenure"] > 24) & (dataframe["tenure"] <= 36), "NEW_TENURE"] = "2-3 Year"
    dataframe.loc[(dataframe["tenure"] > 36) & (dataframe["tenure"] <= 48), "NEW_TENURE"] = "3-4 Year"
    dataframe.loc[(dataframe["tenure"] > 48) & (dataframe["tenure"] <= 60), "NEW_TENURE"] = "4-5 Year"
    dataframe.loc[(dataframe["tenure"] > 60) & (dataframe["tenure"] <= 72), "NEW_TENURE"] = "5-6 Year"

    # Create a new feature 'NEW_AVG_CHARGES' representing average charges per month
    dataframe["NEW_AVG_CHARGES"] = dataframe["TotalCharges"] / (dataframe["tenure"])

    # Create a new feature 'NEW_INCREASE' representing the increase in charges
    dataframe["NEW_INCREASE"] = dataframe["NEW_AVG_CHARGES"] / dataframe["MonthlyCharges"]

    # Designate customers with 1 or 2-year contracts as 'Engaged'
    dataframe["NEW_Engaged"] = dataframe["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

    # Identify customers who do not have any support, backup, or protection services
    dataframe["NEW_noProt"] = dataframe.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

    # Create a new feature 'NEW_Young_Not_Engaged' for monthly contract and young customers
    dataframe["NEW_Young_Not_Engaged"] = dataframe.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

    # Calculate the total number of services subscribed by each customer
    dataframe['NEW_TotalServices'] = (dataframe[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

    # Create a new feature 'NEW_FLAG_ANY_STREAMING' indicating if a customer has any streaming service
    dataframe["NEW_FLAG_ANY_STREAMING"] = dataframe.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

    # Create a new feature 'NEW_FLAG_AutoPayment' indicating if a customer pays automatically
    dataframe["NEW_FLAG_AutoPayment"] = dataframe["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)


    # Calculate the average service fee per service subscribed
    dataframe["NEW_AVG_Service_Fee"] = dataframe["MonthlyCharges"] / (dataframe['NEW_TotalServices'] + 1)

    # Identify customers with fiber optic internet service
    dataframe["new_FiberOptic"] = dataframe.apply(lambda x: 1 if (x["InternetService"] == "Fiber optic") else 0, axis=1)

    # Identify customers who conduct online transactions
    dataframe["new_OnlineTransaction"] = dataframe.apply(lambda x: 1 if (x["PaperlessBilling"] == "Yes") or ((x["PaymentMethod"] == "Electronic check") or (x["PaymentMethod"] == "Mailed check")) else 0, axis=1)

    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


