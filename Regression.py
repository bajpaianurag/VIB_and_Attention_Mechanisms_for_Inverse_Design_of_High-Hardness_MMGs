import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import shap
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

# Load dataset
data = pd.read_csv('H_v_dataset.csv')
X = data.drop(['Load','HV'], axis=1).values
y = data['HV'].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparamter Space and Model Initiation
regressor_models = {
    "RF": (RandomForestRegressor(), {
        'n_estimators': (10, 200),
        'max_depth': (2, 30),
        'min_samples_split': (2, 30),
        'min_samples_leaf': (1, 30),
        'max_features': (0.1, 1.0),
        'bootstrap': [True, False],
        'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
        'min_impurity_decrease': (0.0, 0.1),  
        'min_weight_fraction_leaf': (0.0, 0.5),  
        'max_leaf_nodes': (10, 50),
        'ccp_alpha': (0.0, 0.2)
    }),
    "Lasso": (Lasso(), {
        'alpha': (0.000001, 1.0),                     
        'selection': ['cyclic', 'random']
    }),
    "Ridge": (Ridge(), {
        'alpha': (0.000001, 10.0),                     
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }),
    "KNN": (KNeighborsRegressor(), {
        'n_neighbors': (2, 100),                      
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }),
    "GB": (GradientBoostingRegressor(), {
        'n_estimators': (10, 200),
        'learning_rate': (0.00001, 1.0),              
        'max_depth': (1, 30),                        
        'min_samples_split': (2, 30),                               
        'max_features': (0.1, 1.0),  
    })
}


## Modelling HV
results = {}
best_models = {}

for name, (model, search_space) in regressor_models.items():
    opt = BayesSearchCV(model, search_space, n_iter=200, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    opt.fit(X_train, y_train)
    best_models[name] = opt.best_estimator_

    y_train_pred = opt.predict(X_train)
    y_test_pred = opt.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    
    results[name] = {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'mse': mse,
        'r2': r2,
        'mae': mae
    }
    
    print(f"{name} MSE: {mse:.4f}")
    print(f"{name} R2 Score: {r2:.4f}")
    print(f"{name} MAE: {mae:.4f}")

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test.flatten(), y_test_pred.flatten(), s=100, alpha=0.6)
    plt.plot([min(y_test.flatten()), max(y_test.flatten())], [min(y_test.flatten()), max(y_test.flatten())], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot for {name}')
    plt.legend()
    plt.savefig(f'{name}_parity_plot_HV.png')
    plt.show()

# Save datasets and results to Excel
with pd.ExcelWriter('regression_results_HV_without_load.xlsx') as writer:
    pd.DataFrame(X_train).to_excel(writer, sheet_name='X_train', index=False)
    pd.DataFrame(y_train, columns=['HV']).to_excel(writer, sheet_name='y_train', index=False)
    pd.DataFrame(X_test).to_excel(writer, sheet_name='X_test', index=False)
    pd.DataFrame(y_test, columns=['HV']).to_excel(writer, sheet_name='y_test', index=False)

    for name, result in results.items():
        df_train_pred = pd.DataFrame(result['y_train_pred'], columns=['HV_train_pred'])
        df_test_pred = pd.DataFrame(result['y_test_pred'], columns=['HV_test_pred'])
        df_train_pred.to_excel(writer, sheet_name=f'{name}_train_predictions', index=False)
        df_test_pred.to_excel(writer, sheet_name=f'{name}_test_predictions', index=False)

    metrics_df = pd.DataFrame([
        {'Model': name, 'MSE': result['mse'], 'R2 Score': result['r2'], 'MAE': result['mae']}
        for name, result in results.items()
    ])
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

print("Regression results saved to 'regression_results.xlsx'.")



