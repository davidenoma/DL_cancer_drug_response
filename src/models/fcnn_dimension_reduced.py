"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Fully-connected model for PCA dimension-reduced data
Author(s): Mojtaba Kanani Sarcheshmeh
Source: dl_cancer_drug_response.ipynb cells 217-309.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Imports hoisted from earlier notebook cells
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Attention, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

# Fully Connected Model for Dimension-Reduced Data (Mojtaba)

# Additional Data Preprocessing

# Initially, my plan was to utilize the dataset containing all the gene expressions. Given the comprehensive nature of this dataset, I decided to conduct a thorough examination of its features. This detailed analysis was crucial to understand the underlying data structure and to identify any potential patterns or issues that might influence our model's performance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from IPython.display import FileLink

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#data_1000 = pd.read_csv('/kaggle/input/601-28-project/final_1000/content/data/final_1000.csv').drop('model_id', axis=1)
data_all = pd.read_csv('/kaggle/input/601-28-project/final_all/content/data/final_all.csv').drop('model_id', axis=1)
data_all = pd.get_dummies(data_all, columns=['SANGER_MODEL_ID', 'DRUG_NAME'], dtype=int)
data_all = data_all[list(data_all.columns[3:])+list(data_all.columns[:3])]

# To prevent any data leakage, I first split the data using a 60-20-20 ratio. This specific ratio was chosen due to the relatively small sample size in comparison to the large number of features.

def train_val_test_split(data, train_size=0.6, val_size=0.2, test_size=0.2, random_state=seed_value):
    """
    Split the data into train, validation, and test sets.
    
    Parameters:
    - data: DataFrame or numpy array, the dataset to split.
    - train_size: float, proportion of the dataset to include in the train split (default: 0.6).
    - val_size: float, proportion of the dataset to include in the validation split (default: 0.2).
    - test_size: float, proportion of the dataset to include in the test split (default: 0.2).
    - random_state: int or RandomState instance, controls the shuffling applied to the data (default: None).
    
    Returns:
    - train_data: DataFrame or numpy array, training set.
    - val_data: DataFrame or numpy array, validation set.
    - test_data: DataFrame or numpy array, test set.
    """
    # Split into train and rest
    train_data, rest_data = train_test_split(data, train_size=train_size, random_state=random_state, shuffle=True)
    
    # Calculate remaining sizes after train split
    remaining_size = 1.0 - train_size
    val_relative_size = val_size / remaining_size
    
    # Split rest into validation and test
    val_data, test_data = train_test_split(rest_data, train_size=val_relative_size, random_state=random_state, shuffle=True)
    
    return train_data, val_data, test_data

#del val_data_all
#del train_data_all
#del test_data_all

#train_data_1000, val_data_1000, test_data_1000 = train_val_test_split(data_1000)
train_data_all, val_data_all, test_data_all = train_val_test_split(data_all)

def plot_cumulative_variance(df, min_cumulative=0.99):
    # Calculate variance for each feature
    variances = df.var(axis=0)

    # Sort the variances in descending order
    sorted_variances = np.sort(variances)[::-1]

    # Calculate the cumulative sum of the sorted variances
    cumulative_variances = np.cumsum(sorted_variances)

    # Normalize the cumulative variances
    cumulative_variances_normalized = cumulative_variances / cumulative_variances[-1]

    # Find the earliest index where the cumulative variance is higher than `min_cumulative`
    index_99 = np.argmax(cumulative_variances_normalized > min_cumulative)

    # Plot the cumulative variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variances_normalized, label='Cumulative Variance')

    # Add a point for the 99% explained variance
    plt.scatter(index_99, cumulative_variances_normalized[index_99], color='red', zorder=5)
    plt.axhline(y=0.99, color='gray', linestyle='--', label='99% Variance Explained')
    plt.axvline(x=index_99, color='gray', linestyle='--')
    plt.text(index_99 + 5, 0.95, f'Index: {index_99}', color='red', fontsize=12)

    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained by Sorted Features')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower center')
    plt.show()

# This plot displays the sorted cumulative explained variance for the 1024 features representing the drug chemical. As illustrated, we need 871 features to explain 99 percent of the variance in the data.

DrugChem_start_idx = list(train_data_all.columns).index('DrugChem_1')
DrugChem_end_idx = list(train_data_all.columns).index('DrugChem_1024')+1

plot_cumulative_variance(train_data_all.iloc[:,DrugChem_start_idx:DrugChem_end_idx])

# For the gene expression data, we found that using only 3686 features allows us to explain 99 percent of the variance, instead of utilizing all 38,293 features. Although this represents a significant reduction, 3686 features still constitute a substantial amount, which could pose challenges for model training. Therefore, I decided to apply Principal Component Analysis (PCA) to further decrease the number of features, aiming to make the training process more manageable and efficient.

GeneExp_start_idx = list(train_data_all.columns).index('A1BG')
GeneExp_end_idx = list(train_data_all.columns).index('ATP6V1FNB')+1

plot_cumulative_variance(train_data_all.iloc[:,GeneExp_start_idx:GeneExp_end_idx])

def plot_pca_cumulative_variance(df):
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA()
    pca.fit(df_standardized)

    # Get the explained variance ratios
    explained_variance_ratios = pca.explained_variance_ratio_

    # Calculate the cumulative variance explained
    cumulative_explained_variance = np.cumsum(explained_variance_ratios)

    # Find the earliest index where the cumulative variance is higher than 0.99
    index_99 = np.argmax(cumulative_explained_variance > 0.99)

    # Plot the cumulative variance explained by the principal components
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_explained_variance, label='Cumulative Variance')

    # Add a point for the 99% explained variance
    plt.scatter(index_99, cumulative_explained_variance[index_99], color='red', zorder=5)
    plt.axhline(y=0.99, color='gray', linestyle='--', label='99% Variance Explained')
    plt.axvline(x=index_99, color='gray', linestyle='--')
    plt.text(index_99 + 1, 0.90, f'Index: {index_99}', color='red', fontsize=12)

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained by Principal Components')
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower center')
    plt.show()
    
    return scaler, pca, index_99

# After applying Principal Component Analysis (PCA) to the drug chemical encodings data, I managed to reduce the number of features significantly. By using only 111 principal components, I was able to explain 99 percent of the variance, compared to the 871 features originally needed. This reduction greatly simplifies the data, making the subsequent model training process more efficient and less computationally intensive.

scaler_DrugChem, pca_DrugChem, index_99_DrugChem = plot_pca_cumulative_variance(train_data_all.iloc[:,DrugChem_start_idx:DrugChem_end_idx])

# After applying Principal Component Analysis (PCA) to the gene expression data, I achieved a significant reduction in the number of features. Using only 57 principal components, I was able to explain 99 percent of the variance, compared to the 3686 features originally required. This substantial reduction simplifies the dataset, making it much more manageable for model training and improving computational efficiency.

scaler_GeneExp, pca_GeneExp, index_99_GeneExp = plot_pca_cumulative_variance(train_data_all.iloc[:,GeneExp_start_idx:GeneExp_end_idx])

def scale_pca(df, scaler, pca, clipping_idx=-1, column_prefix='PC'):
    scale_transformed = scaler.transform(df)
    scale_pca_transformed = pca.transform(scale_transformed)
    component_names = [f'{column_prefix}{i+1}' for i in range(pca.n_components_)]
    scale_pca_transformed_df = pd.DataFrame(data=scale_pca_transformed, columns=component_names)
    scale_pca_transformed_clipped_df = scale_pca_transformed_df.iloc[:,:clipping_idx if clipping_idx>0 else None]
    scale_pca_transformed_clipped_df.index = df.index
    
    return scale_pca_transformed_clipped_df

def scale(df, scaler):
    scale_transformed = scaler.transform(df)
    scale_transformed_df = pd.DataFrame(scale_transformed, columns=df.columns)
    scale_transformed_df.index = df.index
    
    return scale_transformed_df

def replace_columns(df, columns_to_drop_start_idx, columns_to_drop_end_idx, df_to_concat):
    df = df.drop(df.columns[columns_to_drop_start_idx:columns_to_drop_end_idx], axis=1)
    return pd.concat((df, df_to_concat), axis=1)

# At this stage, I used the training data to fit both the scaling and PCA transformers. As previously mentioned, I applied PCA separately to the drug chemical encodings and the gene expression data. For the remaining features, I performed normalization. After fitting the transformers on the training data, I then transformed the test and validation datasets to ensure there was no data leakage.

DrugChem_start_idx = list(train_data_all.columns).index('DrugChem_1')
DrugChem_end_idx = list(train_data_all.columns).index('DrugChem_1024')+1

train_data_all = replace_columns(df = train_data_all,
                                 columns_to_drop_start_idx = DrugChem_start_idx,
                                 columns_to_drop_end_idx = DrugChem_end_idx,
                                 df_to_concat = scale_pca(df = train_data_all.iloc[:,DrugChem_start_idx:DrugChem_end_idx], 
                                                          scaler = scaler_DrugChem, 
                                                          pca = pca_DrugChem, 
                                                          clipping_idx = index_99_DrugChem, 
                                                          column_prefix = 'DrugChemPCA_'))


GeneExp_start_idx = list(train_data_all.columns).index('A1BG')
GeneExp_end_idx = list(train_data_all.columns).index('ATP6V1FNB')+1

train_data_all = replace_columns(df = train_data_all,
                                 columns_to_drop_start_idx = GeneExp_start_idx,
                                 columns_to_drop_end_idx = GeneExp_end_idx,
                                 df_to_concat = scale_pca(df = train_data_all.iloc[:,GeneExp_start_idx:GeneExp_end_idx], 
                                                          scaler = scaler_GeneExp, 
                                                          pca = pca_GeneExp, 
                                                          clipping_idx = index_99_GeneExp, 
                                                          column_prefix = 'GeneExpPCA_'))


Rest_start_idx = list(train_data_all.columns).index('SANGER_MODEL_ID_SIDM00046')
Rest_end_idx = list(train_data_all.columns).index('MAX_CONC')+1

scaler_Rest = StandardScaler()
scaler_Rest.fit(train_data_all.iloc[:,Rest_start_idx:Rest_end_idx])

train_data_all = replace_columns(df = train_data_all,
                                 columns_to_drop_start_idx = Rest_start_idx,
                                 columns_to_drop_end_idx = Rest_end_idx,
                                 df_to_concat = scale(df = train_data_all.iloc[:,Rest_start_idx:Rest_end_idx], 
                                                      scaler = scaler_Rest))

print(train_data_all.shape)
train_data_all.head()

DrugChem_start_idx = list(test_data_all.columns).index('DrugChem_1')
DrugChem_end_idx = list(test_data_all.columns).index('DrugChem_1024')+1

test_data_all = replace_columns(df = test_data_all,
                                 columns_to_drop_start_idx = DrugChem_start_idx,
                                 columns_to_drop_end_idx = DrugChem_end_idx,
                                 df_to_concat = scale_pca(df = test_data_all.iloc[:,DrugChem_start_idx:DrugChem_end_idx], 
                                                          scaler = scaler_DrugChem, 
                                                          pca = pca_DrugChem, 
                                                          clipping_idx = index_99_DrugChem, 
                                                          column_prefix = 'DrugChemPCA_'))


GeneExp_start_idx = list(test_data_all.columns).index('A1BG')
GeneExp_end_idx = list(test_data_all.columns).index('ATP6V1FNB')+1

test_data_all = replace_columns(df = test_data_all,
                                 columns_to_drop_start_idx = GeneExp_start_idx,
                                 columns_to_drop_end_idx = GeneExp_end_idx,
                                 df_to_concat = scale_pca(df = test_data_all.iloc[:,GeneExp_start_idx:GeneExp_end_idx], 
                                                          scaler = scaler_GeneExp, 
                                                          pca = pca_GeneExp, 
                                                          clipping_idx = index_99_GeneExp, 
                                                          column_prefix = 'GeneExpPCA_'))


Rest_start_idx = list(test_data_all.columns).index('SANGER_MODEL_ID_SIDM00046')
Rest_end_idx = list(test_data_all.columns).index('MAX_CONC')+1

test_data_all = replace_columns(df = test_data_all,
                                 columns_to_drop_start_idx = Rest_start_idx,
                                 columns_to_drop_end_idx = Rest_end_idx,
                                 df_to_concat = scale(df = test_data_all.iloc[:,Rest_start_idx:Rest_end_idx], 
                                                      scaler = scaler_Rest))

print(test_data_all.shape)
test_data_all.head()

DrugChem_start_idx = list(val_data_all.columns).index('DrugChem_1')
DrugChem_end_idx = list(val_data_all.columns).index('DrugChem_1024')+1

val_data_all = replace_columns(df = val_data_all,
                                 columns_to_drop_start_idx = DrugChem_start_idx,
                                 columns_to_drop_end_idx = DrugChem_end_idx,
                                 df_to_concat = scale_pca(df = val_data_all.iloc[:,DrugChem_start_idx:DrugChem_end_idx], 
                                                          scaler = scaler_DrugChem, 
                                                          pca = pca_DrugChem, 
                                                          clipping_idx = index_99_DrugChem, 
                                                          column_prefix = 'DrugChemPCA_'))


GeneExp_start_idx = list(val_data_all.columns).index('A1BG')
GeneExp_end_idx = list(val_data_all.columns).index('ATP6V1FNB')+1

val_data_all = replace_columns(df = val_data_all,
                                 columns_to_drop_start_idx = GeneExp_start_idx,
                                 columns_to_drop_end_idx = GeneExp_end_idx,
                                 df_to_concat = scale_pca(df = val_data_all.iloc[:,GeneExp_start_idx:GeneExp_end_idx], 
                                                          scaler = scaler_GeneExp, 
                                                          pca = pca_GeneExp, 
                                                          clipping_idx = index_99_GeneExp, 
                                                          column_prefix = 'GeneExpPCA_'))


Rest_start_idx = list(val_data_all.columns).index('SANGER_MODEL_ID_SIDM00046')
Rest_end_idx = list(val_data_all.columns).index('MAX_CONC')+1

val_data_all = replace_columns(df = val_data_all,
                                 columns_to_drop_start_idx = Rest_start_idx,
                                 columns_to_drop_end_idx = Rest_end_idx,
                                 df_to_concat = scale(df = val_data_all.iloc[:,Rest_start_idx:Rest_end_idx], 
                                                      scaler = scaler_Rest))

print(val_data_all.shape)
val_data_all.head()

train_data_all.to_csv('train_pca.csv', index=False)
test_data_all.to_csv('test_pca.csv', index=False)
val_data_all.to_csv('val_pca.csv', index=False)

# After these preprocessing steps, the number of features was reduced to just 352, while retaining a significant amount of the variance in the data.

# Hyperparameter tuning and Evaluation

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
import random 
import os

from sklearn.linear_model import LinearRegression
from scipy import stats
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback, History
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error
from tensorflow.keras.optimizers import Adam

train = pd.read_csv('/kaggle/input/601-28-project-datapca/train_pca.csv')
test = pd.read_csv('/kaggle/input/601-28-project-datapca/test_pca.csv')
val = pd.read_csv('/kaggle/input/601-28-project-datapca/val_pca.csv')

X_train, y_train = train.drop('LN_IC50', axis=1), train['LN_IC50']
X_test,  y_test  = test.drop('LN_IC50', axis=1),  test['LN_IC50']
X_val,   y_val   = val.drop('LN_IC50', axis=1),   val['LN_IC50']

# Verify if GPU is being used
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Done")
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def set_seed(seed=42):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def evaluation(true, predicted, title='Evaluation'):
    print(f"{title}")
    print(f"- MSE : {mean_squared_error(true, predicted)}")
    print(f"- R2 : {r2_score(true, predicted)}")
    print('')

# To establish a baseline for comparing the performance of more complex models, I initially trained a simple linear regression on the dataset. Given that validation data is unnecessary for creating a basic linear regression model, I combined the training and validation datasets for training purposes. Here are the performance results of the model:

reg = LinearRegression().fit(np.concatenate((X_train, X_val), axis=0), 
                             np.concatenate((y_train, y_val), axis=0))

evaluation(y_train, list(reg.predict(X_train)), title="train")
evaluation(y_test,  list(reg.predict(X_test)),  title="test")
evaluation(y_val,   list(reg.predict(X_val)),   title="val")

# I decided to use a fully connected neural network (FCNN) structure because in my opinion this dataset doesn't have any temporal or spatial properties that would benefit from recurrent neural networks (RNNs) or convolutional neural networks (CNNs). FCNNs are ideal for tabular data like this, where each sample is independent and features are not related in a sequential or spatial manner.

# I opted for stacking multiple dense layers with batch normalization and dropout regularization because this approach allows the model to learn hierarchical representations of features. Batch normalization helps in stabilizing and accelerating the training process by normalizing the inputs to each layer. Dropout regularization prevents overfitting by randomly dropping connections between layers during training, ensuring that the model doesn't rely too heavily on specific features.

# To find the best model configuration, I systematically tuned hyperparameters such as the number of layers, units per layer, dropout ratio, and learning rate using Keras Tuner. This iterative process ensures that the model is finely tuned to achieve optimal performance metrics, particularly minimizing mean squared error (MSE) for my regression task.

# I evaluated the models based on their performance on a separate validation dataset. Early stopping based on validation loss helped me prevent overfitting and identify the optimal number of training epochs, ensuring that the model generalizes well to new, unseen data.

# In summary, the FCNN structure with batch normalization, dropout regularization, and systematic hyperparameter tuning was chosen to maximize model performance and generalization capability for this specific dataset characteristics and modeling goals.

# I initially attempted methods like random search and Bayesian estimation to find the optimal parameter combinations for my model. However, lacking clear intuition on the task's complexity and which parameters to focus on, I shifted my approach to gain a deeper understanding of how different parameter settings affect model performance.

# I began with a simple model comprising a few nodes in one layer and closely monitored the training and validation loss per epoch. At each step, I iteratively adjusted the model based on observed trends:
# - Addressing Underfitting: When encountering underfitting, I enhanced the model complexity by:
#   - Adding additional layers to deepen the neural network.
#   - Increasing the number of units per layer to allow for more complex feature representation.
#   - Decreasing the dropout ratio to reduce regularization effects.
#   - Increasing the batch size to facilitate more stable updates during training.

# - Addressing Overfitting: In response to overfitting, I simplified the model by:
#   - Removing unnecessary layers to reduce model complexity.
#   - Decreasing the number of units per layer to limit the capacity of the model.
#   - Increasing the dropout ratio to enhance regularization and prevent the model from memorizing noise in the training data.
#   - Decreasing the batch size to introduce more stochasticity and prevent the model from fitting too closely to the training data.

# - Managing Unstable Performance: If the model's performance showed instability, I adjusted by:
#   - Reducing the learning rate to allow for more gradual updates and smoother convergence.

# Through this iterative process of adjusting model parameters based on observed performance metrics, I aimed to strike a balance between underfitting and overfitting while maximizing the model's predictive capability. This methodical approach allowed me to gain insights into how each parameter setting influences the model's behavior and performance, ultimately guiding the refinement of the model architecture for improved results.

def model_structure_1(num_layers_par, num_units_par, dropout_ratio_par, learning_rate_par, num_epochs, batch_size, patience=100):
    set_seed()
    
    tuner_dir = 'my_dir/kt'
    log_dir = 'my_dir/logs'
    os.makedirs(tuner_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Check if there's only one combination in the parameter grid
    if len(num_layers_par) == 1 and len(num_units_par) == 1 and len(dropout_ratio_par) == 1 and len(learning_rate_par) == 1:
        num_layers = num_layers_par[0]
        num_units = num_units_par[0]
        dropout_ratio = dropout_ratio_par[0]
        learning_rate = learning_rate_par[0]

        print(f"Using fixed hyperparameters: num_layers={num_layers}, num_units={num_units}, dropout={dropout_ratio}, learning_rate={learning_rate}")

        # Build the model
        model = Sequential()
        
        # Input layer
        model.add(Dense(units=num_units, activation='relu', input_shape=(352,)))
        model.add(BatchNormalization()) 
        model.add(Dropout(dropout_ratio))

        # Hidden layers
        for i in range(num_layers):
            model.add(Dense(units=num_units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_ratio))

        # Output layer
        model.add(Dense(1))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)])

        # Evaluate the model
        print("Train Evaluation:")
        train_loss = model.evaluate(X_train, y_train)
        print("Test Evaluation:")
        test_loss = model.evaluate(X_test, y_test)
        print("Validation Evaluation:")
        val_loss = model.evaluate(X_val, y_val)

        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        return history, None, model  # Return None for tuner object when not using Keras Tuner

    else:
        # Function to build the model
        def build_model(hp):
            model = Sequential()
            
            num_layers = hp.Choice('num_layers', num_layers_par)
            num_units = hp.Choice('num_units', num_units_par)
            dropout_ratio = hp.Choice('dropout', dropout_ratio_par)
            learning_rate = hp.Choice('learning_rate', learning_rate_par)

            # Input layer
            model.add(Dense(units=num_units, activation='relu', input_shape=(352,)))
            model.add(BatchNormalization()) 
            model.add(Dropout(dropout_ratio))

            # Hidden layers
            for i in range(num_layers):
                model.add(Dense(units=num_units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_ratio))

            # Output layer
            model.add(Dense(1))

            # Compile model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

            return model

        # Instantiate the tuner
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=20,
            executions_per_trial=1,
            directory=tuner_dir,
            project_name='intro_to_kt',
            overwrite=True,
        )

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        
        # Run the tuner
        tuner.search(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val),
                     callbacks=[early_stopping])

        # Get the best model
        best_trial = tuner.oracle.get_best_trials(1)[0]
        best_model = tuner.hypermodel.build(best_trial.hyperparameters)  # Build the model from the best trial's hyperparameters

        # Compile the best model
        best_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

        # Train the best model again on the entire dataset
        history = best_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Evaluate the best model
        print("Train Evaluation:")
        train_loss = best_model.evaluate(X_train, y_train)
        print("Test Evaluation:")
        test_loss = best_model.evaluate(X_test, y_test)
        print("Validation Evaluation:")
        val_loss = best_model.evaluate(X_val, y_val)

        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        return history, tuner, best_model

history, tuner, best_model = model_structure_1(num_layers_par=[1], 
                                               num_units_par=[5], 
                                               dropout_ratio_par=[0.0], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=10)

history, tuner, best_model = model_structure_1(num_layers_par=[2], 
                                               num_units_par=[5], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=100)

history, tuner, best_model = model_structure_1(num_layers_par=[5], 
                                               num_units_par=[5], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[5], 
                                               num_units_par=[5], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.05], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[5], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.05], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[20], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.05], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[20], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.1], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[20], 
                                               dropout_ratio_par=[0.3], 
                                               learning_rate_par=[0.1], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[64], 
                                               dropout_ratio_par=[0.3], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[64], 
                                               dropout_ratio_par=[0.5], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[15], 
                                               num_units_par=[64], 
                                               dropout_ratio_par=[0.4], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[20], 
                                               num_units_par=[64], 
                                               dropout_ratio_par=[0.3], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=200)

history, tuner, best_model = model_structure_1(num_layers_par=[20], 
                                               num_units_par=[100], 
                                               dropout_ratio_par=[0.3], 
                                               learning_rate_par=[0.01], 
                                               num_epochs=400)

history, tuner, best_model = model_structure_1(num_layers_par=[20], 
                                               num_units_par=[100], 
                                               dropout_ratio_par=[0.35], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400)

history, tuner, best_model = model_structure_1(num_layers_par=[20], 
                                               num_units_par=[100], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[15], 
                                               num_units_par=[100], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[100], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[80], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[10], 
                                               num_units_par=[60], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[5], 
                                               num_units_par=[80], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[5], 
                                               num_units_par=[200], 
                                               dropout_ratio_par=[0.45], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[5], 
                                               num_units_par=[200], 
                                               dropout_ratio_par=[0.6], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

history, tuner, best_model = model_structure_1(num_layers_par=[5], 
                                               num_units_par=[200], 
                                               dropout_ratio_par=[0.7], 
                                               learning_rate_par=[0.02], 
                                               num_epochs=400, 
                                               patience=200)

# The best model I discovered through this iterative method achieved an MSE of 1.21. However, a recurring issue observed across these models is the challenge of overfitting. Attempts to mitigate overfitting by increasing the dropout ratio often resulted in the model struggling to achieve a low MSE even on the training data, indicating signs of underfitting.

# To address this issue, I explored a new model structure by incorporating kernel regularizers. The goal was to leverage these regularization techniques alongside dropout to potentially enhance the model's ability to generalize while maintaining low training and validation errors.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError

def build_model(input_size, hidden_sizes, output_size, learning_rate=0.001,
                activation='relu', dropout_rate=0.2, kernel_regularizer=None,
                batch_normalization=False, optimizer='adam', loss='mse', batch_size=32):
    
    set_seed()
    
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_sizes[0], input_dim=input_size, activation=activation,
                    kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer))
    
    if batch_normalization:
        model.add(BatchNormalization())
    
    # Hidden layers
    for size in hidden_sizes[1:]:
        model.add(Dense(size, activation=activation, kernel_regularizer=kernel_regularizer))
        model.add(Dropout(dropout_rate))
        if batch_normalization:
            model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(output_size, activation='linear'))  # Linear activation for regression
    
    # Select optimizer
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer}' not supported.")
    
    # Select loss function
    if loss == 'mse':
        loss_function = MeanSquaredError()
    elif loss == 'huber':
        loss_function = Huber()
    else:
        raise ValueError(f"Loss function '{loss}' not supported.")
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])
    
    return model

input_size = 352
hidden_sizes = [256, 128, 64]  # hidden layer sizes
output_size = 1

# Hyperparameters
learning_rate = 0.001
activation = 'relu'
dropout_rate = 0.2
kernel_regularizer = None  # l2(0.01)
batch_normalization = True  # Set to True to use Batch Normalization
loss = 'mse'  # Options: 'mse'
batch_size = 64 

set_seed()
# Build the model
model = build_model(input_size, hidden_sizes, output_size, learning_rate=learning_rate,
                    activation=activation, dropout_rate=dropout_rate,
                    kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization,
                    optimizer=optimizer, loss=loss)

# Display model architecture
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=150, verbose=1),  # Early stopping with patience of 5 epochs
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),  # Save best model weights
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, min_lr=1e-6, verbose=1)  # Reduce learning rate on plateau
]


# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=400,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Load the best weights
model.load_weights('best_model.keras')

# Evaluate the best model
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Hyperparameters
learning_rate = 0.001
activation = 'relu'
dropout_rate = 0.2
kernel_regularizer = l2(0.01)  #l2(0.01)
batch_normalization = True  # Set to True to use Batch Normalization
loss = 'mse'  # Options: 'mse'
batch_size = 128
epochs = 400

set_seed()
# Build the model
model = build_model(input_size, hidden_sizes, output_size, learning_rate=learning_rate,
                    activation=activation, dropout_rate=dropout_rate,
                    kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization,
                    optimizer=optimizer, loss=loss)

# Display model architecture
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=150, verbose=1),  # Early stopping with patience of 5 epochs
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),  # Save best model weights
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, min_lr=1e-6, verbose=1)  # Reduce learning rate on plateau
]


# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Load the best weights
model.load_weights('best_model.keras')

# Evaluate the best model
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

input_size = 352
hidden_sizes = [256, 128, 64, 32]  # Example hidden layer sizes
output_size = 1

# Hyperparameters
learning_rate = 0.001
activation = 'relu'
dropout_rate = 0.2
kernel_regularizer = l2(0.005)
batch_normalization = True
loss = 'mse'
batch_size = 128
epochs = 400

set_seed()
# Build the model
model = build_model(input_size, hidden_sizes, output_size, learning_rate=learning_rate,
                    activation=activation, dropout_rate=dropout_rate,
                    kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization,
                    optimizer=optimizer, loss=loss)

# Display model architecture
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=150, verbose=1),  # Early stopping with patience of 150 epochs
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),  # Save best model weights
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, min_lr=1e-6, verbose=1)  # Reduce learning rate on plateau
]


# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Load the best weights
model.load_weights('best_model.keras')

# Evaluate the best model
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Adjusting hyperparameters for further experimentation
hidden_sizes = [128, 64, 32]
kernel_regularizer = l2(0.001)
dropout_rate = 0.3

model = build_model(input_size, hidden_sizes, output_size,
                    learning_rate=learning_rate, activation=activation,
                    dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization, optimizer=optimizer, loss=loss)

epochs = 600
callbacks = [
    EarlyStopping(monitor='val_loss', patience=200, verbose=1),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Evaluate the best model
model.load_weights('best_model.keras')
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

input_size = 352
hidden_sizes = [128, 64, 32]
output_size = 1

learning_rate = 0.001
activation = 'relu'
dropout_rate = 0.4
kernel_regularizer = l2(0.0001)
batch_normalization = True
optimizer = 'adam'
loss = 'mse'
batch_size = 128
epochs = 600


model = build_model(input_size, hidden_sizes, output_size,
                    learning_rate=learning_rate, activation=activation,
                    dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization, optimizer=optimizer, loss=loss)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=200, verbose=1),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Evaluate the best model
model.load_weights('best_model.keras')
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

input_size = 352
hidden_sizes = [128, 64, 32]
output_size = 1

learning_rate = 0.001
activation = 'relu'
dropout_rate = 0.5  # Increased dropout rate
kernel_regularizer = l2(0.01)  # Increased regularization strength
batch_normalization = True
optimizer = 'adam'
loss = 'mse'
batch_size = 128
epochs = 600

model = build_model(input_size, hidden_sizes, output_size,
                    learning_rate=learning_rate, activation=activation,
                    dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization, optimizer=optimizer, loss=loss)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=200, verbose=1),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Evaluate the best model
model.load_weights('best_model.keras')
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

input_size = 352
hidden_sizes = [64, 32]  # Simplified model architecture
output_size = 1

learning_rate = 0.001
activation = 'relu'
dropout_rate = 0.3  # Fine-tuned dropout rate
kernel_regularizer = l2(0.01)  # Keep regularization strength high
batch_normalization = True
optimizer = 'adam'
loss = 'mse'
batch_size = 64  # Smaller batch size
epochs = 800

model = build_model(input_size, hidden_sizes, output_size,
                    learning_rate=learning_rate, activation=activation,
                    dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization, optimizer=optimizer, loss=loss)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=200, verbose=1),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Evaluate the best model
model.load_weights('best_model.keras')
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

input_size = 352
hidden_sizes = [64, 32]  # Simplified model architecture
output_size = 1

learning_rate = 0.0001
activation = 'relu'
dropout_rate = 0.6  # Increased dropout rate
kernel_regularizer = l2(0.1)  # Increased regularization strength
batch_normalization = True
optimizer = 'adam'
loss = 'huber'  # Changed to Huber loss
batch_size = 64  # Smaller batch size
epochs = 800


model = build_model(input_size, hidden_sizes, output_size,
                    learning_rate=learning_rate, activation=activation,
                    dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer,
                    batch_normalization=batch_normalization, optimizer=optimizer, loss=loss)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=200, verbose=1),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# Evaluate the best model
model.load_weights('best_model.keras')
print("Train Evaluation:")
train_loss = model.evaluate(X_train, y_train)
print("Test Evaluation:")
test_loss = model.evaluate(X_test, y_test)
print("Validation Evaluation:")
val_loss = model.evaluate(X_val, y_val)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# I also explored models that treat drug-related features and gene expression data separately from the rest of the features. The rationale behind this approach is to allow the model to learn more specialized weights for each type of feature, potentially improving its ability to capture distinct patterns and relationships within each dataset subset.

# By processing drug-related features and gene expression data separately, the model can focus on the unique characteristics and dependencies within these subsets. This segregation can lead to more targeted feature representations and more effective learning of relevant information specific to each type of data.

# The investigation involved designing separate branches or pathways within the neural network architecture for drug-related features, gene expression data, and other features. Each pathway could then have its own set of layers, activations, and regularization techniques tailored to the characteristics of its respective feature set.

# This approach aims to optimize the model's performance by allowing it to extract and utilize the most pertinent information from each data subset independently. By enhancing the model's ability to discern and process diverse types of information effectively, it seeks to mitigate issues such as overfitting and improve overall predictive accuracy.

def model_structure_3(num_layers_par, num_units_par, dropout_ratio_par, learning_rate_par, num_epochs, batch_size=32, patience=100):
    set_seed()
    
    tuner_dir = 'my_dir/kt'
    log_dir = 'my_dir/logs'
    os.makedirs(tuner_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model_checkpoint_path = os.path.join(log_dir, "best_model.h5")

    # Define the cosine decay schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate_par[0],
        decay_steps=num_epochs * (len(X_train) // batch_size)
    )

    # Inputs
    drug_input = Input(shape=(111,))
    gene_input = Input(shape=(57,))
    general_input = Input(shape=(X_train.shape[1] - 111 - 57,))

    # Drug subnetwork
    drug_net = Dense(units=num_units_par[0], activation='relu')(drug_input)
    drug_net = BatchNormalization()(drug_net)
    drug_net = Dropout(dropout_ratio_par[0])(drug_net)
    for _ in range(num_layers_par[0] - 1):
        drug_net = Dense(units=num_units_par[0], activation='relu')(drug_net)
        drug_net = BatchNormalization()(drug_net)
        drug_net = Dropout(dropout_ratio_par[0])(drug_net)

    # Gene subnetwork
    gene_net = Dense(units=num_units_par[0], activation='relu')(gene_input)
    gene_net = BatchNormalization()(gene_net)
    gene_net = Dropout(dropout_ratio_par[0])(gene_net)
    for _ in range(num_layers_par[0] - 1):
        gene_net = Dense(units=num_units_par[0], activation='relu')(gene_net)
        gene_net = BatchNormalization()(gene_net)
        gene_net = Dropout(dropout_ratio_par[0])(gene_net)

    # General subnetwork
    general_net = Dense(units=num_units_par[0], activation='relu')(general_input)
    general_net = BatchNormalization()(general_net)
    general_net = Dropout(dropout_ratio_par[0])(general_net)
    for _ in range(num_layers_par[0] - 1):
        general_net = Dense(units=num_units_par[0], activation='relu')(general_net)
        general_net = BatchNormalization()(general_net)
        general_net = Dropout(dropout_ratio_par[0])(general_net)

    # Concatenate
    combined = Concatenate()([drug_net, gene_net, general_net])

    # Attention mechanism
    attention_output = Dense(num_units_par[0], activation='relu')(combined)
    attention_output = Attention()([combined, combined])
    attention_output = LayerNormalization()(attention_output)

    # Final layers after attention
    for _ in range(num_layers_par[0]):
        attention_output = Dense(units=num_units_par[0], activation='relu')(attention_output)
        attention_output = BatchNormalization()(attention_output)
        attention_output = Dropout(dropout_ratio_par[0])(attention_output)

    # Output layer
    output = Dense(1)(attention_output)

    # Compile model
    model = Model(inputs=[drug_input, gene_input, general_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule), loss='mean_squared_error')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit([X_train[:, :111], X_train[:, 111:168], X_train[:, 168:]], y_train,
                        epochs=num_epochs, batch_size=batch_size, validation_data=([X_val[:, :111], X_val[:, 111:168], X_val[:, 168:]], y_val),
                        callbacks=[early_stopping, model_checkpoint])

    # Load the best weights
    model.load_weights(model_checkpoint_path)

    # Evaluate the model
    print("Train Evaluation:")
    train_loss = model.evaluate([X_train[:, :111], X_train[:, 111:168], X_train[:, 168:]], y_train)
    print("Test Evaluation:")
    test_loss = model.evaluate([X_test[:, :111], X_test[:, 111:168], X_test[:, 168:]], y_test)
    print("Validation Evaluation:")
    val_loss = model.evaluate([X_val[:, :111], X_val[:, 111:168], X_val[:, 168:]], y_val)

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return history, None, model

history, tuner, best_model = model_structure_3(num_layers_par=[5],
                                               num_units_par=[50],
                                               dropout_ratio_par=[0.2],
                                               learning_rate_par=[0.001],
                                               num_epochs=400,
                                               patience=200,
                                               batch_size=32)

history, tuner, best_model = model_structure_3(num_layers_par=[5],
                                               num_units_par=[50],
                                               dropout_ratio_par=[0.4],
                                               learning_rate_par=[0.001],
                                               num_epochs=400,
                                               patience=200,
                                               batch_size=32)

history, tuner, best_model = model_structure_3(num_layers_par=[5],
                                               num_units_par=[50],
                                               dropout_ratio_par=[0.4],
                                               learning_rate_par=[0.001],
                                               num_epochs=400,
                                               patience=200,
                                               batch_size=32)

history, tuner, best_model = model_structure_3(num_layers_par=[5],
                                               num_units_par=[50],
                                               dropout_ratio_par=[0.4],
                                               learning_rate_par=[0.001],
                                               num_epochs=400,
                                               patience=200,
                                               batch_size=64)

# Among all the models I experimented with, the best performance I achieved was using the initial model structure, which resulted in an MSE of 1.21.
