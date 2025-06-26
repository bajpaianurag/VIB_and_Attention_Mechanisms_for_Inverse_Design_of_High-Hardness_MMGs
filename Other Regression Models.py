import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import os

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


# Build up the MLP model
def build_mlp(hidden_layers=2, hidden_units=64, learning_rate=0.001, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(X_train.shape[1],), activation=activation))
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(1))

    opt = {
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate)
    }.get(optimizer, Adam(learning_rate=learning_rate))

    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


# Modelling HV
results = {}
best_models = {}

for name, (model, search_space) in regressor_models.items():
    print(f"Training {name}...")
    opt = BayesSearchCV(model, search_space, n_iter=10, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    opt.fit(X_train, y_train)
    best_models[name] = opt.best_estimator_
    
    # Store the best hyperparameters
    best_hyperparams = opt.best_params_
    
    # Predictions
    y_train_pred = opt.predict(X_train)
    y_test_pred = opt.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    
    results[name] = {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'best_hyperparams': best_hyperparams
    }
    
    print(f"{name} MSE: {mse:.4f}")
    print(f"{name} R2 Score: {r2:.4f}")
    print(f"{name} MAE: {mae:.4f}")
    print(f"Best Hyperparameters for {name}: {best_hyperparams}\n")

    # Plot Parity Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test.flatten(), y_test_pred.flatten(), s=100, alpha=0.6)
    plt.plot([min(y_test.flatten()), max(y_test.flatten())], [min(y_test.flatten()), max(y_test.flatten())], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot for {name}')
    plt.legend()
    plt.savefig(f'{name}_parity_plot_HV.png')
    plt.show()


# Learning curves
def plot_learning_curves_detailed(estimator, X, y, title='', cv=5, train_sizes=np.linspace(0.1, 1.0, 5), model_name='model', suffix=''):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='neg_mean_squared_error', train_sizes=train_sizes, return_times=False
    )
    train_rmse = np.sqrt(-train_scores)
    test_rmse = np.sqrt(-test_scores)
    train_mae_scores = []
    test_mae_scores = []

    for i, frac in enumerate(train_sizes):
        mae_train_folds, mae_test_folds = [], []
        for train_idx, val_idx in list(cv.split(X, y)):
            X_train_frac = X[train_idx[:int(frac*len(train_idx))]]
            y_train_frac = y[train_idx[:int(frac*len(train_idx))]]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            est = clone(estimator)
            est.fit(X_train_frac, y_train_frac)
            pred_train = est.predict(X_train_frac)
            pred_val = est.predict(X_val_fold)
            mae_train_folds.append(mean_absolute_error(y_train_frac, pred_train))
            mae_test_folds.append(mean_absolute_error(y_val_fold, pred_val))
        train_mae_scores.append(mae_train_folds)
        test_mae_scores.append(mae_test_folds)

    train_mae_scores = np.array(train_mae_scores)
    test_mae_scores = np.array(test_mae_scores)

    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, np.mean(train_rmse, axis=1), 'o-', label="Train RMSE", color='red')
    plt.fill_between(train_sizes, np.mean(train_rmse, axis=1) - np.std(train_rmse, axis=1),
                     np.mean(train_rmse, axis=1) + np.std(train_rmse, axis=1), color='red', alpha=0.2)
    plt.plot(train_sizes, np.mean(test_rmse, axis=1), 'o-', label="Val RMSE", color='green')
    plt.fill_between(train_sizes, np.mean(test_rmse, axis=1) - np.std(test_rmse, axis=1),
                     np.mean(test_rmse, axis=1) + np.std(test_rmse, axis=1), color='green', alpha=0.2)
    plt.xlabel("Training Samples")
    plt.ylabel("RMSE")
    plt.title(f"{title} ({suffix}) - RMSE")
    plt.legend()
    plt.grid(True)

cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in best_models.items():
    plot_learning_curves_detailed(model, X, y, title=name, cv=cv_kfold, model_name=name, suffix='kfold')

