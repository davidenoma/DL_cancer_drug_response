"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Cross-validation and hyperparameter tuning
Author(s): Ariel Ghislain Kemogne Kamdoum
Source: dl_cancer_drug_response.ipynb cells 197-206.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Cross-Validation and Hyperparameter Tuning (Ariel)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, LSTM
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '/content/sample_data/final_drug_modelid_minmaxconc_unique.csv'
data = pd.read_csv(file_path)

# Encode the 'model_id' and 'drug' columns
label_encoder = LabelEncoder()
data['model_id'] = label_encoder.fit_transform(data['model_id'])
data['drug'] = label_encoder.fit_transform(data['drug'])

# Drop columns that contain non-numeric values explicitly
data = data.select_dtypes(include=[np.number])

# Handle missing values
data = data.fillna(data.mean())

# Split data into features and target
X = data.drop(columns=['LN_IC50'])
y = data['LN_IC50']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest with Hyperparameter Tuning

# Define the Random Forest model with hyperparameter tuning
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search with cross-validation
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)

# Evaluate the model
best_rf = grid_search_rf.best_estimator_
rf_predictions = best_rf.predict(X_test_scaled)
print(f'Random Forest - Best Parameters: {grid_search_rf.best_params_}')
print(f'Random Forest - RMSE: {np.sqrt(mean_squared_error(y_test, rf_predictions))}, R2: {r2_score(y_test, rf_predictions)}')

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Function to plot learning curves
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves for Random Forest
title = "Learning Curves (Random Forest)"
plot_learning_curve(best_rf, title, X_train_scaled, y_train, cv=5, n_jobs=-1)
plt.show()

# Multi Linear Regression with Cross-Validation

# Define the Multi Linear Regression model
mlr = LinearRegression()

# Perform cross-validation
cv_scores_mlr = cross_val_score(mlr, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
mlr.fit(X_train_scaled, y_train)
mlr_predictions = mlr.predict(X_test_scaled)

print(f'Multi Linear Regression - Cross-Validation RMSE: {np.sqrt(-cv_scores_mlr.mean())}')
print(f'Multi Linear Regression - RMSE: {np.sqrt(mean_squared_error(y_test, mlr_predictions))}, R2: {r2_score(y_test, mlr_predictions)}')

# Multi-Layer Perceptron with Hyperparameter Tuning

# Define the MLP model with hyperparameter tuning
mlp = MLPRegressor(random_state=42)
param_grid_mlp = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [500, 1000]
}

# Perform grid search with cross-validation
grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_mlp.fit(X_train_scaled, y_train)

# Evaluate the model
best_mlp = grid_search_mlp.best_estimator_
mlp_predictions = best_mlp.predict(X_test_scaled)
print(f'MLP - Best Parameters: {grid_search_mlp.best_params_}')
print(f'MLP - RMSE: {np.sqrt(mean_squared_error(y_test, mlp_predictions))}, R2: {r2_score(y_test, mlp_predictions)}')

import matplotlib.pyplot as plt

# Train a Multi-Layer Perceptron model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

history = mlp_model.fit(X_train_scaled, y_train)

# Plot training loss
plt.figure()
plt.plot(history.loss_curve_, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MLP Training Loss')
plt.legend()
plt.show()

# Evaluate the model
mlp_predictions = mlp_model.predict(X_test_scaled)
print(f'MLP - RMSE: {np.sqrt(mean_squared_error(y_test, mlp_predictions))}, R2: {r2_score(y_test, mlp_predictions)}')
