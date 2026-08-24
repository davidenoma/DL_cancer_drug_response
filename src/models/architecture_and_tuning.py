"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Model architecture and hyperparameter tuning (RF, MLR, MLP, CNN)
Author(s): David Enoma
Source: dl_cancer_drug_response.ipynb cells 81-92.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# pip install rdkit keras_tuner

# Model Architecture (David)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from keras.src.layers import Flatten, Conv1D, MaxPooling1D
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras_tuner import GridSearch, BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from kerastuner.tuners import Hyperband

# Function to downcast data types to save memory
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype == 'int64']
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    return df

# Load RNAseq data and calculate variances
chunk_size = 10000
rna_seq_variances = pd.Series(dtype='float32')
chunk_count = 0

for chunk in pd.read_csv('transposed_filtered_rnaseq_read_count_all.csv', chunksize=chunk_size):
    chunk = downcast_dtypes(chunk)
    chunk_count += 1
    chunk_data = chunk.drop(columns=['model_id'])
    chunk_variances = chunk_data.var(axis=0)
    rna_seq_variances = rna_seq_variances.add(chunk_variances, fill_value=0)

rna_seq_variances /= chunk_count

top_n = 1000
top_genes = rna_seq_variances.nlargest(top_n).index
top_genes_df = pd.DataFrame(top_genes, columns=['gene'])
top_genes_df.to_csv('top_genes.csv', index=False)

top_genes = pd.read_csv('top_genes.csv')
top_genes_list = top_genes['gene'].tolist()

reduced_rna_seq_df = pd.DataFrame()

for chunk in pd.read_csv('transposed_filtered_rnaseq_read_count_all.csv', chunksize=chunk_size):
    chunk = downcast_dtypes(chunk)
    reduced_chunk = chunk[['model_id'] + top_genes_list]
    reduced_rna_seq_df = pd.concat([reduced_rna_seq_df, reduced_chunk], ignore_index=True)

reduced_rna_seq_df.to_csv('reduced_rnaseq_read_count_all.csv', index=False)

# Load drug response data
drug_response_df = pd.read_csv('merged_drug_response_with_smiles_no_na.csv')

# Example SMILES encoding using Morgan fingerprints (you can adjust as needed)
def smiles_to_morgan_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fingerprint)
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return None

# Apply SMILES encoding to 'smile' column
drug_response_df['Morgan_Fingerprint'] = drug_response_df['smile'].apply(smiles_to_morgan_fingerprint)

# Ensure all fingerprints are valid and not None
drug_response_df = drug_response_df.dropna(subset=['Morgan_Fingerprint'])

# Stack the numpy arrays into a 2D array for scaling
fingerprint_matrix = np.stack(drug_response_df['Morgan_Fingerprint'].values)

# Optionally, you can apply StandardScaler to the fingerprint data if needed
scaler = StandardScaler()
fingerprint_matrix_scaled = scaler.fit_transform(fingerprint_matrix)

# Add scaled fingerprints back to the DataFrame as separate columns
fingerprint_columns = [f'fingerprint_{i}' for i in range(fingerprint_matrix_scaled.shape[1])]
fingerprint_df = pd.DataFrame(fingerprint_matrix_scaled, columns=fingerprint_columns)
drug_response_df = pd.concat([drug_response_df.reset_index(drop=True), fingerprint_df.reset_index(drop=True)], axis=1)

# Drop the original unscaled fingerprint column
drug_response_df.drop(columns=['Morgan_Fingerprint'], inplace=True)

# Perform one-hot encoding (dummy encoding) for categorical variables if needed
# Here we assume that smile column was categorical before being used for fingerprints and you may want to keep it
drug_response_df_encoded = pd.get_dummies(drug_response_df, columns=['smile'], drop_first=True)

# Print the dataframe to check results
print(drug_response_df.head())

# Save the dataframe if needed
drug_response_df.to_csv('encoded_drug_response.csv', index=False)

# Prepare merged dataset with RNAseq and drug response data
merged_df = pd.merge(drug_response_df, reduced_rna_seq_df, left_on='SANGER_MODEL_ID', right_on='model_id')

X = merged_df.drop(columns=['SANGER_MODEL_ID', 'DRUG_NAME', 'LN_IC50', 'model_id','smile'])
y = merged_df['LN_IC50']
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Random Forest Regressor ###
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R² Score: {r2_rf}')

joblib.dump(rf_model, 'rf_model.joblib')

### Multiple Linear Regression (MLR) ###
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

y_pred_mlr = mlr_model.predict(X_test)
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)
print(f'Multiple Linear Regression Mean Squared Error: {mse_mlr}')
print(f'Multiple Linear Regression R² Score: {r2_mlr}')

# Mulilayer Perceptron

import tensorflow as tf
import random
import numpy as np
# Set seeds for reproducibility
random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def build_mlp_model(hp):
    model = Sequential()

    # Input layer
    model.add(Dense(units=hp.Int('units_input', min_value=64, max_value=256, step=32),
                    activation='relu',
                    kernel_initializer=hp.Choice('kernel_initializer_input', values=['glorot_uniform', 'he_normal']),
                    input_shape=(X_train.shape[1],)))

    # Number of hidden layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=5)
    for i in range(num_layers):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                        activation='relu',
                        kernel_initializer=hp.Choice(f'kernel_initializer_{i}', values=['glorot_uniform', 'he_normal'])))
        model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1))

    # Learning rate for Adam optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mean_squared_error')

    return model
#
# tuner = GridSearch(
#     build_mlp_model,
#     objective='val_loss',
#     max_trials=50,
#     overwrite=True,
#     directory='keras_tuner_logs',
#     project_name='mlp_hyperparameter_tuning'
# )
tuner = BayesianOptimization(
    build_mlp_model,
    objective='val_loss',
    max_trials=10,
    directory='my_dir',
    overwrite=True,
    project_name='intro_to_kt'
)

tuner.search(X_train_scaled, y_train, epochs=50, validation_split=0.2)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hp)

history = best_model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=1)

# Predict and evaluate the optimized MLP model
y_pred_best_mlp = best_model.predict(X_test_scaled)


mse_best_mlp = mean_squared_error(y_test, y_pred_best_mlp)
r2_best_mlp = r2_score(y_test, y_pred_best_mlp)
mae_best_mlp = mean_absolute_error(y_test, y_pred_best_mlp)
print(f'Optimized MLP Mean Squared Error: {mse_best_mlp}')
print(f'Optimized MLP R² Score: {r2_best_mlp}')
print(f'Optimized MLP Mean Absolute Error: ',{mae_best_mlp})


# Optionally, plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('MLP Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

from keras.utils import plot_model
# Plot the architecture of the best CNN model
plot_model(best_model, to_file='best_model_architecture.png', show_shapes=True)
# Note: Save the best model and scaler if needed
# best_model.save('best_mlp_model_tf')
# joblib.dump(scaler, 'scaler.joblib')

# Convolutional Neural Network
#  The hyperparameter optimization of the CNN involved systematically tuning filters, kernel sizes, initializers, dense units, dropout rates, and learning rates, resulting in a model with a Mean Squared Error of 1.1598 and an R² score of 0.8316, demonstrating strong predictive capability for drug sensitivity.

import pandas as pd
import numpy as np
from keras.src.layers import Flatten, Conv1D, MaxPooling1D
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras_tuner import GridSearch, BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from kerastuner.tuners import Hyperband
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers.legacy import Adam
from kerastuner.tuners import BayesianOptimization
import numpy as np

import numpy as np
import random

def build_large_cnn(hp):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv1D(filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=16),
                     kernel_size=hp.Int('conv1_kernel', min_value=3, max_value=7, step=2),
                     activation='relu',
                     kernel_initializer=hp.Choice('conv1_init', values=['he_normal', 'glorot_uniform']),
                     input_shape=(X_train_scaled.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Second Convolutional Block
    model.add(Conv1D(filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
                     kernel_size=hp.Int('conv2_kernel', min_value=3, max_value=7, step=2),
                     activation='relu',
                     kernel_initializer=hp.Choice('conv2_init', values=['he_normal', 'glorot_uniform'])))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Third Convolutional Block
    model.add(Conv1D(filters=hp.Int('conv3_filters', min_value=128, max_value=512, step=64),
                     kernel_size=hp.Int('conv3_kernel', min_value=3, max_value=7, step=2),
                     activation='relu',
                     kernel_initializer=hp.Choice('conv3_init', values=['he_normal', 'glorot_uniform'])))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))



    model.add(Flatten())

    # First Dense Block
    model.add(Dense(units=hp.Int('dense1_units', min_value=64, max_value=512, step=32),
                    activation='relu',
                    kernel_initializer=hp.Choice('dense1_init', values=['he_normal', 'glorot_uniform'])))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout1_rate', min_value=0.1, max_value=0.5, step=0.1)))

    # Second Dense Block
    model.add(Dense(units=hp.Int('dense2_units', min_value=64, max_value=512, step=32),
                    activation='relu',
                    kernel_initializer=hp.Choice('dense2_init', values=['he_normal', 'glorot_uniform'])))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout2_rate', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')

    return model

# Setup Keras Tuner BayesianOptimization
tuner = BayesianOptimization(
    build_large_cnn,
    objective='val_loss',
    max_trials=10,  # Adjust as needed
    directory='tuner_dir',
    project_name='large_cnn_bayesian_opt'
)

# Perform hyperparameter search
tuner.search(X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1),
             y_train,
             epochs=50,
             validation_split=0.2,
             verbose=1)

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best CNN model with the optimal hyperparameters
best_cnn_model = tuner.hypermodel.build(best_hp)

# Train the best CNN model
history_best_cnn = best_cnn_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1),
                                      y_train,
                                      epochs=100,
                                      batch_size=64,
                                      validation_split=0.2,
                                      verbose=1)

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best CNN model with the optimal hyperparameters
best_cnn_model = tuner.hypermodel.build(best_hp)

# Train the best CNN model
history_best_cnn = best_cnn_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1),
                                      y_train,
                                      epochs=100,
                                      batch_size=64,
                                      validation_split=0.2,
                                      verbose=1)

# Evaluate the best CNN model
y_pred_best_cnn = best_cnn_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
mse_best_cnn = mean_squared_error(y_test, y_pred_best_cnn)
r2_best_cnn = r2_score(y_test, y_pred_best_cnn)
mae_best_cnn = mean_absolute_error(y_test, y_pred_best_cnn)
print(f'Optimized CNN Mean Squared Error: {mse_best_cnn}')
print(f'Optimized CNN R² Score: {r2_best_cnn}')
print(f'Optimized CNN Mean Absolute Error: {mae_best_cnn}')

# Optionally, plot the loss curve for best CNN model
plt.figure(figsize=(10, 6))
plt.plot(history_best_cnn.history['loss'], label='train_loss')
plt.plot(history_best_cnn.history['val_loss'], label='val_loss')
plt.title('Optimized CNN Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

from keras.utils import plot_model
# Plot the architecture of the best CNN model
plot_model(best_cnn_model, to_file='best_cnn_model_architecture.png', show_shapes=True)
