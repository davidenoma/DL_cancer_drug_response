"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Model investigation: baselines, neural network, CNN and deep NN
Author(s): Ariel Ghislain Kemogne Kamdoum
Source: dl_cancer_drug_response.ipynb cells 59-80.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Imports hoisted from earlier notebook cells
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Models

# Models Investigation (Ariel)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN

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

# Random forest

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
rf_predictions = rf_model.predict(X_test_scaled)
print(f'Random Forest - RMSE: {np.sqrt(mean_squared_error(y_test, rf_predictions))}, R2: {r2_score(y_test, rf_predictions)}')

# Multi-linear regression

# Train a Multi Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
mlr_predictions = mlr_model.predict(X_test_scaled)
print(f'Multi Linear Regression - RMSE: {np.sqrt(mean_squared_error(y_test, mlr_predictions))}, R2: {r2_score(y_test, mlr_predictions)}')

# Multi-Layer Perceptron

# Train a Multi-Layer Perceptron model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
mlp_predictions = mlp_model.predict(X_test_scaled)
print(f'Multi-Layer Perceptron - RMSE: {np.sqrt(mean_squared_error(y_test, mlp_predictions))}, R2: {r2_score(y_test, mlp_predictions)}')

# Neural network

# Build a basic Neural Network model
nn_model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(50, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Neural Network model
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions and evaluate the model
nn_predictions = nn_model.predict(X_test_scaled).flatten()
print(f'Neural Network - RMSE: {np.sqrt(mean_squared_error(y_test, nn_predictions))}, R2: {r2_score(y_test, nn_predictions)}')

# Convolution neural network

# Reshape data for CNN
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Build a CNN model
cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

cnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions and evaluate the model
cnn_predictions = cnn_model.predict(X_test_cnn).flatten()
print(f'CNN - RMSE: {np.sqrt(mean_squared_error(y_test, cnn_predictions))}, R2: {r2_score(y_test, cnn_predictions)}')

# Train the CNN model
history_cnn = cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Plot training and validation loss for CNN
plt.figure()
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Training and Validation Loss')
plt.legend()
plt.show()

# Deep Neural Network

# The task is to predict IC50 values for drug responses using gene expression data. The part is designed of a neural network model that processes gene expression data through an encoder and then uses a fully connected predictor to output the IC50 values.

# Model Architecture
# The model is designed based on the provided diagrams, which include an encoder for gene expression data and a fully connected predictor.

# Gene Expression Encoder:
# Input Layer: The input layer accepts the gene expression data with a shape equal to the number of features in the training set.

# Hidden Layers:
# The first hidden layer has 512 neurons with ReLU activation. ReLU (Rectified Linear Unit) is a commonly used activation function that helps in mitigating the vanishing gradient problem.

# A Dropout layer with a rate of 0.3 is added to prevent overfitting by randomly setting 30% of the input units to 0 during training.

# The second hidden layer has 256 neurons with ReLU activation, followed by another Dropout layer with a 0.3 rate.

# The third hidden layer has 128 neurons with ReLU activation. This layer represents the encoded gene expression features.
# Fully Connected Predictor:

# Concatenation: The encoded features from the gene expression encoder are concatenated (although in this implementation, it only includes the gene expression features).

# Dense Layers:
# The first dense layer in the predictor has 128 neurons with ReLU activation, followed by a Dropout layer with a 0.3 rate.
# The second dense layer also has 128 neurons with ReLU activation, followed by another Dropout layer with a 0.3 rate.

# Output Layer: The output layer has a single neuron with no activation function to predict the IC50 value. Since this is a regression task, no activation function is applied here.

# Model Training and Evaluation
# Model Compilation:
# The model is compiled using the Adam optimizer, which is an adaptive learning rate optimization algorithm that's widely used in deep learning.

# The loss function is mean squared error (MSE), appropriate for regression tasks where the goal is to minimize the squared differences between predicted and actual values.

# Mean absolute error (MAE) is used as an additional metric to evaluate the performance of the model.

# Training:
# The model is trained for 100 epochs with a batch size of 32. An epoch is one complete pass through the training dataset.
# 20% of the training data is used for validation during training to monitor the model's performance on unseen data and prevent overfitting.

# Evaluation:
# After training, the model's performance is evaluated on the test set using MAE. The test set helps in assessing how well the model generalizes to new, unseen data.

# Summary

# This implementation effectively combines biological knowledge (gene expression data) with deep learning techniques to predict IC50 values for drug responses. The model architecture ensure that it is both biologically relevant and computationally efficient. By focusing on normalization, dropout for regularization, and a robust neural network design, the model aims to achieve high performance in predicting drug responses.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.utils import plot_model



# Define the gene expression encoder
gene_expression_input = Input(shape=(X_train.shape[1],), name='gene_expression_input')
x = Dense(512, activation='relu')(gene_expression_input)
x = Dropout(0.1)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
gene_expression_encoded = Dense(128, activation='relu', name='gene_expression_encoded')(x)

# Combine encoded features
combined = concatenate([gene_expression_encoded])

# Fully connected predictor
y = Dense(128, activation='relu')(combined)
y = Dropout(0.1)(y)
y = Dense(128, activation='relu')(y)
y = Dropout(0.1)(y)
output = Dense(1, name='output')(y)

# Define the model
model = Model(inputs=[gene_expression_input], outputs=[output])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Visualize the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#  Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)

# Print the evaluation results
print(f'Test MAE: {test_mae:.4f}')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print the MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse:.4f}')

#Plotting loss and accuracy
plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model MAE')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Multi layer perceptron
