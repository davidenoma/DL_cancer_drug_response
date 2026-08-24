"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Explainable AI (XAI) with SHAP - PCA and autoencoder
Author(s): Ariel Ghislain Kemogne Kamdoum
Source: dl_cancer_drug_response.ipynb cells 207-214.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Imports hoisted from earlier notebook cells
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Explainable AI (XAI) with SHAP (Ariel)

# Explanation:
# DeepExplainer: This SHAP explainer is used for deep learning models. It helps to interpret the neural network's predictions.

# shap_values: Calculated SHAP values for the test set. These values indicate the contribution of each feature to the model's predictions.

# Visualizations:
# Force Plot: Shows the contribution of each feature to a single prediction. This plot helps to understand the impact of individual features.

# Summary Plot: Provides a global interpretation by showing the SHAP values for all features across all predictions. It highlights the most important features and their impact on the predictions.

# Visualization Details

# Force Plot:
# Visualizes the contribution of each feature to a single prediction. The base value (expected value) is the average model output over the training dataset. Each feature's contribution is shown as pushing the prediction higher or lower.

# Summary Plot:
# Displays a summary of feature importance across the dataset. Each point represents a SHAP value for a feature and an instance. The color indicates the feature value (red for high, blue for low).

# !pip install shap

# SHAP with PCA

import shap
from sklearn.decomposition import PCA

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=20)  # Adjust the number of components to 20
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

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
grid_search_mlp.fit(X_train_pca, y_train)

# Evaluate the model
best_mlp = grid_search_mlp.best_estimator_
mlp_predictions = best_mlp.predict(X_test_pca)
print(f'MLP - Best Parameters: {grid_search_mlp.best_params_}')
print(f'MLP - RMSE: {np.sqrt(mean_squared_error(y_test, mlp_predictions))}, R2: {r2_score(y_test, mlp_predictions)}')

# Use SHAP to explain the MLP model
explainer = shap.KernelExplainer(best_mlp.predict, X_train_pca[:100])  # Using a subset for the background data
shap_values = explainer.shap_values(X_test_pca, nsamples=100)

# Plot the SHAP summary
shap.summary_plot(shap_values, X_test_pca, feature_names=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

# Plot the SHAP dependence plot for the most important feature
most_important_feature = np.argmax(np.abs(shap_values).mean(0))
shap.dependence_plot(most_important_feature, shap_values, X_test_pca, feature_names=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

# Explanation of SHAP Outputs

# The outputs include two types of SHAP visualizations: a summary plot and a dependence plot. Let's delve into what each of these plots tells us about our model and its predictions.

# Summary Plot
# The summary plot is a comprehensive visualization of feature importance and their impacts on the model's output.

# Feature Importance:
# The features are listed on the y-axis in descending order of their importance. In our plot, PC10, PC8, PC9, etc., are the most important features.

# The importance of each feature is determined by the mean absolute value of the SHAP values for that feature across all samples.
# SHAP Values:
# The x-axis represents the SHAP value, which indicates the impact of that feature on the prediction. Positive SHAP values push the prediction higher, while negative SHAP values push it lower.

# Each dot represents a SHAP value for a particular instance. The spread of dots along the x-axis for a single feature shows how varied the impact of that feature is across different instances.

# Feature Value:
# The color of the dots represents the feature value (from low to high, using a blue to red gradient). For example, a red dot for PC10 means that instance had a high value for PC10, while a blue dot means it had a low value.

# The distribution of colors helps to see if high or low feature values are associated with positive or negative impacts on the prediction.

# From the summary plot, we can infer which principal components (PCs) are most influential in our model and how they generally affect the predictions. For example, PC10 has a significant impact, and high values of PC10 (red dots) generally decrease the prediction (negative SHAP values).

# Dependence Plot
# The dependence plot provides a deeper look at how a specific feature (the most important one in this case, PC10) influences the model's prediction while accounting for the effect of another feature (PC7).

# SHAP Values:
# The y-axis represents the SHAP value for PC10. As mentioned, positive values increase the model's prediction, while negative values decrease it.

# Feature Value (PC10):
# The x-axis shows the actual values of PC10.

# Interaction with Another Feature (PC7):
# The color of the dots represents the values of PC7. This helps to visualize how the interaction between PC10 and PC7 affects the prediction.

# In the dependence plot, we can observe:
# As the value of PC10 increases (moving right along the x-axis), the SHAP value generally decreases, indicating that higher PC10 values tend to lower the prediction.

# The interaction with PC7 is also visible: instances with high PC7 values (red dots) might have a different trend compared to those with low PC7 values (blue dots).

# Combined Insights
# Feature Importance:
# PC10 is the most critical feature in our model, followed by PC8, PC9, and so on.

# Impact of PC10:
# Higher values of PC10 generally decrease the model's predictions. This is evident from both the summary and dependence plots.

# Interactions:
# The dependence plot shows the nuanced interaction between PC10 and PC7, suggesting that the relationship between PC10 and the prediction is influenced by the values of PC7.

# Conclusion
# These SHAP plots provide valuable insights into our MLP model's behavior:
# The summary plot highlights which principal components are most important and their overall effect on predictions.

# The dependence plot offers a detailed view of how the most important feature (PC10) interacts with another significant feature (PC7) to influence predictions.

# These visualizations are crucial for understanding model behavior, validating model predictions, and gaining trust in the model, especially when dealing with high-dimensional data reduced via PCA.

# SHAP with autoencoder

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import shap

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the autoencoder
input_dim = X_train_scaled.shape[1]
encoding_dim = 20  # Number of encoded features

input_layer = Input(shape=(input_dim,))
encoded = Dense(100, activation='relu')(input_layer)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Transform the data using the encoder
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

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
grid_search_mlp.fit(X_train_encoded, y_train)

# Evaluate the model
best_mlp = grid_search_mlp.best_estimator_
mlp_predictions = best_mlp.predict(X_test_encoded)
print(f'MLP - Best Parameters: {grid_search_mlp.best_params_}')
print(f'MLP - RMSE: {np.sqrt(mean_squared_error(y_test, mlp_predictions))}, R2: {r2_score(y_test, mlp_predictions)}')

# Use SHAP to explain the MLP model
explainer = shap.KernelExplainer(best_mlp.predict, X_train_encoded[:100])  # Using a subset for the background data
shap_values = explainer.shap_values(X_test_encoded, nsamples=100)

# Plot the SHAP summary
shap.summary_plot(shap_values, X_test_encoded, feature_names=[f'Encoded_Feature{i+1}' for i in range(X_test_encoded.shape[1])])

# Plot the SHAP dependence plot for the most important feature
most_important_feature = np.argmax(np.abs(shap_values).mean(0))
shap.dependence_plot(most_important_feature, shap_values, X_test_encoded, feature_names=[f'Encoded_Feature{i+1}' for i in range(X_test_encoded.shape[1])])
