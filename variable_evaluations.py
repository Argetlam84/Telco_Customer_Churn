import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve




def plot_importance(model, features, num=None, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def plot_importance_for_func(models, features, num=20, save=False):
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)

    for model in models:
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
            plt.title(f"{type(model).__name__} Features")
            plt.tight_layout()
            plt.show()

            if save:
                plt.savefig(f'{type(model).__name__} Features importances.png')
        else:
            print(f"Model {type(model).__name__} does not have feature_importances_. Skipping...")










