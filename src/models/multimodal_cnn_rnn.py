"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Multimodal network: CNN for gene expression + RNN for SMILES
Author(s): Sasha Chernenkoff
Source: dl_cancer_drug_response.ipynb cells 95-195.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Imports hoisted from earlier notebook cells
import time
from tensorflow.keras.layers import Activation, BatchNormalization

# Multimodal RNN & CNN (Sasha)
# Building up the network architecture
# I chose to implement a multimodal CNN & RNN in order to most effectively process the different types of data in our dataset. I used a CNN to process the gene expression data and an RNN to process the encoded SMILES data. The final IC50 prediction also incorporates the remaining data (model id, drug name, and concentration) through a FCNN before making the final IC50 prediction.

# Model architecture
# CNN for gene expression data

# I chose to implement a CNN to process the gene expression data due to the CNNs ability to effectively capture spatial information and patterns in sig-dimensional data. The gene expression data involves tens of thousands of genes, and  CNNs are well-suited for handling this high-dimensional input. The key reasons for selecting CNN include:
# Feature extraction: CNNs are capable of learning and extracting relevant features from the data, which helps in identifying important gene interactions and patterns.

# Spatial invariance: CNNs can recognize patterns regardless of their spatial position in the input, which is beneficial for gene expression data where the relative positioning of gene expression values may vary.

# Computational efficiency: The use of pooling layers in CNNs reduces the dimensionality of the data, leading to more efficient processing and reduced computational complexity.

# RNN for SMILES encodings

# I chose to use an LSTM, a type of RNN, to process the SMILES encodings of the drug compounds. SMILES encodings represent the chemical structures of the drugs as sequences of characters, and I thought an RNN may be able to effectively capture patterns related to the drug structure from this sequential data. I chose to implement an LSTM specifically because the long SMILES encodings contain important information throughout the entire sequence, so maintaining the memory is important. The reasons for choosing RNN include:
# Sequential data handling: RNNs are designed to handle sequential data, so they are well-suited for processing the SMILES encodings where the order of the data is critical for accurately representing chemical structures.

# Memory and context: LSTMs have the ability to maintain long-term dependencies and context, which is important for capturing the information and relationships within the entire drug chemical sequence.

# Pattern recognition: RNNs can learn and recognize patterns within sequences, which allows the model to understand and predict the chemical properties of drugs based on their SMILES encodings.

# Integration and prediction
# The outputs from the CNN processing of the gene expression data and the RNN processing of the SMILES encodings are concatenated along with additional drug information features (drug name, model id, and concentration). This is then passed through a series of fully connected layers to make the final IC50 prediction.

# Data preprocessing
# This involved:
# - Normalizing the numerical features (gene expression, concentrations) and one-hot encoding the categorical features (model id, drug name)
# - Splitting the dataset to feed into different parts of the neural network:
#   - Gene expression data -> CNN
#   - SMILES encodings -> RNN
#   - Remaining drug information -> FCNN for final prediction

# Imports
# !pip install keras-tuner

# import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import tensorflow as tf
import keras_tuner as kt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

### Loading and inspecting data

# The data file consists of a model, drug, min and max conc, gene expression for multiple genes,
# a SMILES of the drug, an encoded vector representing the drug's SMILE, and an IC50 (which is our target)
df = pd.read_csv('final_1000.csv')
df.head()

df.shape

# Model_id is in the dataset twice, so we can drop one
df.drop(columns=['model_id'], inplace=True)
df.shape

# Save the drug SMILES and all the genes as lists
smiles_names = df.columns[5:1029].tolist()
gene_names = df.columns[1029:].tolist()
smiles_names, gene_names

len(smiles_names), len(gene_names)

### Data normalization process and splitting into train/val/test sets

# Investigate how many unqiue drugs there are. We will need a way to
# encode this for our neural network. Similarily, model_id will need to also be encoded
len(df['DRUG_NAME'].unique()), len(df['SANGER_MODEL_ID'].unique())

# I will be using one-hot encoding for the categorical variables (SANGER_MODEL_ID and DRUG_NAME)
# We will fit the OneHotEncoder before splitting the dataset
# One-hot encode 'SANGER_MODEL_ID' and 'DRUG_NAME' columns
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
one_hot_encoder.fit(df[['SANGER_MODEL_ID', 'DRUG_NAME']])

# Now, I'll split the data into training, validation, and test sets
# I'll use an 80/10/10 split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
df_train.shape, df_val.shape, df_test.shape

# Function to apply the one-hot encoding
def apply_one_hot_encoding(df, encoder):
    encoded_categorical = encoder.transform(df[['SANGER_MODEL_ID', 'DRUG_NAME']])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(['SANGER_MODEL_ID', 'DRUG_NAME']))
    df_encoded = pd.concat([df.drop(['SANGER_MODEL_ID', 'DRUG_NAME'], axis=1).reset_index(drop=True), encoded_categorical_df], axis=1)
    return df_encoded

df_train = apply_one_hot_encoding(df_train, one_hot_encoder)
df_val = apply_one_hot_encoding(df_val, one_hot_encoder)
df_test = apply_one_hot_encoding(df_test, one_hot_encoder)
df_train.shape, df_val.shape, df_test.shape

# Now, I will use min max scaling to scale the numerical data
# This will be the MIN_CONC, MAX_CONC, and all genes in gene_names
numerical_cols = ['MIN_CONC', 'MAX_CONC'] + gene_names

# Here, we define the scaler, fit it to the training data and then apply the scaler to
# all the datasets
scaler = MinMaxScaler()

df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])


df_val[numerical_cols] = scaler.transform(df_val[numerical_cols])
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

df_train.shape, df_val.shape, df_test.shape

df_train.head()

# Here, I'm going to figure out the input data sizes for the inputs to different parts of the network

genex_input_size = len(gene_names)
smiles_input_size = len(smiles_names)

total_columns = len(df_train.columns)
drug_info_input_size = total_columns - (genex_input_size + smiles_input_size + 1)

# Define input shapes
genex_input_shape = (genex_input_size,)
smiles_input_shape = (smiles_input_size,)
drug_info_input_shape = (drug_info_input_size,)

df_train.shape, genex_input_shape, smiles_input_shape, drug_info_input_shape

# I'm going to have to split up the datasets to feed to different parts of the network

genex_train = df_train[gene_names]
smiles_train = df_train[smiles_names]
drug_train = df_train.drop(columns=gene_names + smiles_names + ['LN_IC50'])
target_train = df_train['LN_IC50']

genex_val = df_val[gene_names]
smiles_val = df_val[smiles_names]
drug_val = df_val.drop(columns=gene_names + smiles_names + ['LN_IC50'])
target_val = df_val['LN_IC50']

genex_test = df_test[gene_names]
smiles_test = df_test[smiles_names]
drug_test = df_test.drop(columns=gene_names + smiles_names + ['LN_IC50'])
target_test = df_test['LN_IC50']

genex_train.shape, smiles_train.shape, drug_train.shape, target_train.shape

# Setting up the data pipelines with tf.data to improve training speed
batch_size = 32

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((genex_train, smiles_train, drug_train), target_train))
val_dataset = tf.data.Dataset.from_tensor_slices(((genex_val, smiles_val, drug_val), target_val))
test_dataset = tf.data.Dataset.from_tensor_slices(((genex_test, smiles_test, drug_test), target_test))

# Batch and prefetch data
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Bayesian hyperparameter tuning
# Summary of search spaces and best values:
# Datasets:
# - all genes vs. top 1000 genes (best: top 1000)
# - (6474, 2030) vs. (6474, 38293)

# CNN:
# - Num CNN layers tested: 1-8 (best: 7)
# - Kernel size: 3, 5 (best: 3)
# - Pooling size: 2, 3 (best: 2)
# - CNN activation: relu vs. tanh (best: relu)
# - L2 regularization coefficient: 0.01 to 0.0001 (log sampling) (best: 0.0002)
# - Dropout: 0.2 to 0.7 (best: 0.4)

# RNN:
# - Num RNN layers tested: 1-8 (best: 2)
# - LSTM units: 32, 64 (best: 64)
# - L2 regularization: 0.01 to 0.0001 (log sampling) (best: 0.0009)
# - Dropout: 0.2 to 0.7 (best: 0.6)

# FCNN:
# - Num dense layers tested: 1-8 (best: 1)
# - Num hidden units: 64 to 192 (best: 192)
# - Activation: relu, tanh (best: relu)
# - L2 regularization: 0.01 to 0.0001 (log sampling) (best: 0.0001)
# - Dropout: 0.2 to 0.7 (best: 0.2)

# Optimization:
# - Optimizer: adam, rmsprop (best: adam)
# - Learning rate: 0.01 to 0.00001 (log sampling) (best: 0.001)

# Note: hyperparameter tuning was performed in separate notebooks, so I've pasted the code and outputs to the section below.

# I had issues with controlling the size of my model so training wouldn't take an excessively large time. I had to implement a custom callback to monitor the training time at each epoch and skip the trial if the epoch would take a long time to train (> 10 min). Because of this, the first few tuners I ran were having difficulty converging.

# Ultimately, I had to reduce the search space to a maximum of 8 layers at each part of the neural network to keep training to a reasonable timeframe.

# Tuner 1 (all data)

def build_model(hp):
    try:
        print("Building model with hyperparameters:")

        genex_input = Input(shape=genex_input_shape, name='genex_input')
        x = tf.keras.layers.Reshape((genex_input_shape[0], 1))(genex_input)
        num_cnn_layers = hp.Int('num_cnn_layers', 1, 20)
        print(f"Number of CNN layers: {num_cnn_layers}")

        base_filters = hp.Choice('base_filters', values=[32, 64, 128])

        for i in range(num_cnn_layers):
            filters_increment = hp.Choice(f'filters_increment_{i}', values=[4, 8, 12])
            filters = base_filters + filters_increment * i
            kernel_size = hp.Choice(f'cnn_kernel_size_{i}', values=[3, 5])
            activation = hp.Choice(f'cnn_activation_{i}', values=['relu', 'tanh', 'sigmoid'])
            l2_reg = hp.Float(f'cnn_l2_{i}', 0.0001, 0.01, sampling='LOG')
            print(f"CNN layer {i}: filters={filters}, kernel_size={kernel_size}, activation={activation}, l2_reg={l2_reg}")

            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

            # Dynamically adjust pool size based on current input length
            current_length = x.shape[1]
            if current_length is not None:
                max_pool_size = min(3, current_length)
                pool_size = hp.Choice(f'cnn_pool_size_{i}', values=[2, max_pool_size])
                if pool_size > current_length:
                    break
                x = MaxPooling1D(pool_size=pool_size)(x)

        x = Flatten()(x)
        genex_output = Dense(128, activation='relu')(x)

        smiles_input = Input(shape=smiles_input_shape, name='smiles_input')
        y = tf.keras.layers.Reshape((smiles_input_shape[0], 1))(smiles_input)
        num_rnn_layers = hp.Int('num_rnn_layers', 1, 20)
        print(f"Number of RNN layers: {num_rnn_layers}")

        for i in range(num_rnn_layers):
            lstm_units = hp.Choice(f'lstm_units_{i}', values=[32, 64, 128])
            return_sequences = i < (num_rnn_layers - 1)
            lstm_l2_reg = hp.Float(f'lstm_l2_{i}', 0.0001, 0.01, sampling='LOG')
            print(f"RNN layer {i}: units={lstm_units}, return_sequences={return_sequences}, l2_reg={lstm_l2_reg}")

            y = LSTM(units=lstm_units, return_sequences=return_sequences, kernel_regularizer=l2(lstm_l2_reg))(y)

        y = Dense(128, activation='relu')(y)
        smiles_output = y

        concat = Concatenate()([genex_output, smiles_output])
        drug_input = Input(shape=drug_info_input_shape, name='drug_input')
        concat = Concatenate()([concat, drug_input])
        num_dense_layers = hp.Int('num_dense_layers', 1, 20)
        print(f"Number of Dense layers: {num_dense_layers}")

        for i in range(num_dense_layers):
            dense_units = hp.Int(f'dense_units_{i}', min_value=64, max_value=256, step=64)
            dense_activation = hp.Choice(f'dense_activation_{i}', values=['relu', 'tanh', 'sigmoid'])
            dense_l2_reg = hp.Float(f'dense_l2_{i}', 0.0001, 0.01, sampling='LOG')
            dropout_rate = hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)
            print(f"Dense layer {i}: units={dense_units}, activation={dense_activation}, l2_reg={dense_l2_reg}, dropout_rate={dropout_rate}")

            concat = Dense(units=dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2_reg))(concat)
            concat = Dropout(rate=dropout_rate)(concat)

        output = Dense(1)(concat)

        model = Model(inputs=[genex_input, smiles_input, drug_input], outputs=output)

        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
        print(f"Optimizer: {optimizer}")

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )

        return model
    except Exception as e:
        print(f"Error while building model: {e}")
        raise

# Implementing a custom callback to monitor epoch training time. If it excessive, we stop the
# training for this trial and move on.
class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_epoch_duration):
        super(TimeLimitCallback, self).__init__()
        self.max_epoch_duration = max_epoch_duration
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            if epoch_duration > self.max_epoch_duration:
                self.model.stop_training = True
                print(f"\nEpoch exceeded time limit of {self.max_epoch_duration} seconds. Stopping training for this trial.")
                self.epoch_start_time = None

max_epoch_duration = 600  # 10 minutes

# Initialize the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=25,
    executions_per_trial=1,
    directory='tuner_4',
    project_name='bayesian_optimization_large_range'
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Fewer epochs
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        TimeLimitCallback(max_epoch_duration=max_epoch_duration)
    ]
)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test data
test_loss, test_mse = best_model.evaluate(test_dataset)
print(f"Test MSE: {test_mse}")

# Tuner 2 (all data)

def build_model(hp):
    try:
        print("Building model with hyperparameters:")

        genex_input = Input(shape=genex_input_shape, name='genex_input')
        x = tf.keras.layers.Reshape((genex_input_shape[0], 1))(genex_input)
        num_cnn_layers = hp.Int('num_cnn_layers', 1, 8)
        print(f"Number of CNN layers: {num_cnn_layers}")

        base_filters = hp.Choice('base_filters', values=[32, 64])

        for i in range(num_cnn_layers):
            filters_increment = hp.Choice(f'filters_increment_{i}', values=[4, 8])
            filters = base_filters + filters_increment * i
            kernel_size = hp.Choice(f'cnn_kernel_size_{i}', values=[3])
            activation = hp.Choice(f'cnn_activation_{i}', values=['relu', 'tanh'])
            l2_reg = hp.Float(f'cnn_l2_{i}', 0.0001, 0.001, sampling='LOG')
            print(f"CNN layer {i}: filters={filters}, kernel_size={kernel_size}, activation={activation}, l2_reg={l2_reg}")

            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

            current_length = x.shape[1]
            if current_length is not None:
                max_pool_size = min(3, current_length)
                pool_size = hp.Choice(f'cnn_pool_size_{i}', values=[2])
                if pool_size > current_length:
                    break
                x = MaxPooling1D(pool_size=pool_size)(x)

        x = Flatten()(x)
        genex_output = Dense(128, activation='relu')(x)

        smiles_input = Input(shape=smiles_input_shape, name='smiles_input')
        y = tf.keras.layers.Reshape((smiles_input_shape[0], 1))(smiles_input)
        num_rnn_layers = hp.Int('num_rnn_layers', 1, 8)
        print(f"Number of RNN layers: {num_rnn_layers}")

        for i in range(num_rnn_layers):
            lstm_units = hp.Choice(f'lstm_units_{i}', values=[32, 64])
            return_sequences = i < (num_rnn_layers - 1)
            lstm_l2_reg = hp.Float(f'lstm_l2_{i}', 0.0001, 0.001, sampling='LOG')
            print(f"RNN layer {i}: units={lstm_units}, return_sequences={return_sequences}, l2_reg={lstm_l2_reg}")

            y = LSTM(units=lstm_units, return_sequences=return_sequences, kernel_regularizer=l2(lstm_l2_reg))(y)

        y = Dense(128, activation='relu')(y)
        smiles_output = y

        concat = Concatenate()([genex_output, smiles_output])
        drug_input = Input(shape=drug_info_input_shape, name='drug_input')
        concat = Concatenate()([concat, drug_input])
        num_dense_layers = hp.Int('num_dense_layers', 1, 8)
        print(f"Number of Dense layers: {num_dense_layers}")

        for i in range(num_dense_layers):
            dense_units = hp.Int(f'dense_units_{i}', min_value=64, max_value=192, step=64)
            dense_activation = hp.Choice(f'dense_activation_{i}', values=['relu', 'tanh'])
            dense_l2_reg = hp.Float(f'dense_l2_{i}', 0.0001, 0.001, sampling='LOG')
            dropout_rate = hp.Float(f'dropout_{i}', 0.2, 0.4, step=0.1)
            print(f"Dense layer {i}: units={dense_units}, activation={dense_activation}, l2_reg={dense_l2_reg}, dropout_rate={dropout_rate}")

            concat = Dense(units=dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2_reg))(concat)
            concat = Dropout(rate=dropout_rate)(concat)

        output = Dense(1)(concat)

        model = Model(inputs=[genex_input, smiles_input, drug_input], outputs=output)

        optimizer = hp.Choice('optimizer', values=['adam'])
        print(f"Optimizer: {optimizer}")

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )

        return model
    except Exception as e:
        print(f"Error while building model: {e}")
        raise

# Implementing a custom callback to monitor epoch training time. If it excessive, we stop the
# training for this trial and move on.
class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_epoch_duration):
        super(TimeLimitCallback, self).__init__()
        self.max_epoch_duration = max_epoch_duration
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            if epoch_duration > self.max_epoch_duration:
                self.model.stop_training = True
                print(f"\nEpoch exceeded time limit of {self.max_epoch_duration} seconds. Stopping training for this trial.")
                self.epoch_start_time = None

max_epoch_duration = 600  # 10 minutes

# Initialize the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=25,
    executions_per_trial=1,
    directory='tuner_1',
    project_name='bayesian_optimization_large_range'
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Fewer epochs
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        TimeLimitCallback(max_epoch_duration=max_epoch_duration)
    ]
)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test data
test_loss, test_mse = best_model.evaluate(test_dataset)
print(f"Test MSE: {test_mse}")

# Tuner 3 (all data)

def build_model(hp):
    try:
        print("Building model with hyperparameters:")

        genex_input = Input(shape=genex_input_shape, name='genex_input')
        x = tf.keras.layers.Reshape((genex_input_shape[0], 1))(genex_input)
        num_cnn_layers = hp.Int('num_cnn_layers', 1, 8)
        print(f"Number of CNN layers: {num_cnn_layers}")

        base_filters = hp.Choice('base_filters', values=[32, 64])

        for i in range(num_cnn_layers):
            filters_increment = hp.Choice(f'filters_increment_{i}', values=[4, 8])
            filters = base_filters + filters_increment * i
            kernel_size = hp.Choice(f'cnn_kernel_size_{i}', values=[3])
            activation = hp.Choice(f'cnn_activation_{i}', values=['relu', 'tanh'])
            l2_reg = hp.Float(f'cnn_l2_{i}', 0.0001, 0.001, sampling='LOG')
            print(f"CNN layer {i}: filters={filters}, kernel_size={kernel_size}, activation={activation}, l2_reg={l2_reg}")

            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

            current_length = x.shape[1]
            if current_length is not None:
                max_pool_size = min(3, current_length)
                pool_size = hp.Choice(f'cnn_pool_size_{i}', values=[2])
                if pool_size > current_length:
                    break
                x = MaxPooling1D(pool_size=pool_size)(x)

        x = Flatten()(x)
        genex_output = Dense(128, activation='relu')(x)

        smiles_input = Input(shape=smiles_input_shape, name='smiles_input')
        y = tf.keras.layers.Reshape((smiles_input_shape[0], 1))(smiles_input)
        num_rnn_layers = hp.Int('num_rnn_layers', 1, 8)
        print(f"Number of RNN layers: {num_rnn_layers}")

        for i in range(num_rnn_layers):
            lstm_units = hp.Choice(f'lstm_units_{i}', values=[32, 64])
            return_sequences = i < (num_rnn_layers - 1)
            lstm_l2_reg = hp.Float(f'lstm_l2_{i}', 0.0001, 0.001, sampling='LOG')
            print(f"RNN layer {i}: units={lstm_units}, return_sequences={return_sequences}, l2_reg={lstm_l2_reg}")

            y = LSTM(units=lstm_units, return_sequences=return_sequences, kernel_regularizer=l2(lstm_l2_reg))(y)

        y = Dense(128, activation='relu')(y)
        smiles_output = y

        concat = Concatenate()([genex_output, smiles_output])
        drug_input = Input(shape=drug_info_input_shape, name='drug_input')
        concat = Concatenate()([concat, drug_input])
        num_dense_layers = hp.Int('num_dense_layers', 1, 8)
        print(f"Number of Dense layers: {num_dense_layers}")

        for i in range(num_dense_layers):
            dense_units = hp.Int(f'dense_units_{i}', min_value=64, max_value=192, step=64)
            dense_activation = hp.Choice(f'dense_activation_{i}', values=['relu', 'tanh'])
            dense_l2_reg = hp.Float(f'dense_l2_{i}', 0.0001, 0.001, sampling='LOG')
            dropout_rate = hp.Float(f'dropout_{i}', 0.2, 0.4, step=0.1)
            print(f"Dense layer {i}: units={dense_units}, activation={dense_activation}, l2_reg={dense_l2_reg}, dropout_rate={dropout_rate}")

            concat = Dense(units=dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2_reg))(concat)
            concat = Dropout(rate=dropout_rate)(concat)

        output = Dense(1)(concat)

        model = Model(inputs=[genex_input, smiles_input, drug_input], outputs=output)

        optimizer = hp.Choice('optimizer', values=['adam'])
        print(f"Optimizer: {optimizer}")

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )

        return model
    except Exception as e:
        print(f"Error while building model: {e}")
        raise

# Implementing a custom callback to monitor epoch training time. If it excessive, we stop the
# training for this trial and move on.
class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_epoch_duration):
        super(TimeLimitCallback, self).__init__()
        self.max_epoch_duration = max_epoch_duration
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            if epoch_duration > self.max_epoch_duration:
                self.model.stop_training = True
                print(f"\nEpoch exceeded time limit of {self.max_epoch_duration} seconds. Stopping training for this trial.")
                self.epoch_start_time = None

max_epoch_duration = 600  # 10 minutes

# Initialize the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=25,
    executions_per_trial=1,
    directory='tuner_2',
    project_name='bayesian_optimization_large_range'
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Fewer epochs
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        TimeLimitCallback(max_epoch_duration=max_epoch_duration)
    ]
)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test data
test_loss, test_mse = best_model.evaluate(test_dataset)
print(f"Test MSE: {test_mse}")

# Tuner 4 (all data)

def build_model(hp):
    try:
        print("Building model with hyperparameters:")

        genex_input = Input(shape=genex_input_shape, name='genex_input')
        x = tf.keras.layers.Reshape((genex_input_shape[0], 1))(genex_input)
        num_cnn_layers = hp.Int('num_cnn_layers', 1, 8)
        print(f"Number of CNN layers: {num_cnn_layers}")

        base_filters = hp.Choice('base_filters', values=[4, 8, 16])

        for i in range(num_cnn_layers):
            filters = base_filters + 4*i
            kernel_size = hp.Choice(f'cnn_kernel_size_{i}', values=[3])
            activation = hp.Choice(f'cnn_activation_{i}', values=['relu', 'tanh'])
            l2_reg = hp.Float(f'cnn_l2_{i}', 0.0001, 0.001, sampling='LOG')
            print(f"CNN layer {i}: filters={filters}, kernel_size={kernel_size}, activation={activation}, l2_reg={l2_reg}")

            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=None, kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

            current_length = x.shape[1]
            if current_length is not None:
                max_pool_size = min(3, current_length)
                pool_size = hp.Choice(f'cnn_pool_size_{i}', values=[2])
                if pool_size > current_length:
                    break
                x = MaxPooling1D(pool_size=pool_size)(x)

        x = Flatten()(x)
        genex_output = Dense(128, activation='relu')(x)

        smiles_input = Input(shape=smiles_input_shape, name='smiles_input')
        y = tf.keras.layers.Reshape((smiles_input_shape[0], 1))(smiles_input)
        num_rnn_layers = hp.Int('num_rnn_layers', 1, 8)
        print(f"Number of RNN layers: {num_rnn_layers}")

        for i in range(num_rnn_layers):
            lstm_units = hp.Choice(f'lstm_units_{i}', values=[32, 64])
            return_sequences = i < (num_rnn_layers - 1)
            lstm_l2_reg = hp.Float(f'lstm_l2_{i}', 0.0001, 0.001, sampling='LOG')
            print(f"RNN layer {i}: units={lstm_units}, return_sequences={return_sequences}, l2_reg={lstm_l2_reg}")

            y = LSTM(units=lstm_units, return_sequences=return_sequences, kernel_regularizer=l2(lstm_l2_reg))(y)

        y = Dense(128, activation='relu')(y)
        smiles_output = y

        concat = Concatenate()([genex_output, smiles_output])
        drug_input = Input(shape=drug_info_input_shape, name='drug_input')
        concat = Concatenate()([concat, drug_input])
        num_dense_layers = hp.Int('num_dense_layers', 1, 8)
        print(f"Number of Dense layers: {num_dense_layers}")

        for i in range(num_dense_layers):
            dense_units = hp.Int(f'dense_units_{i}', min_value=64, max_value=192, step=64)
            dense_activation = hp.Choice(f'dense_activation_{i}', values=['relu', 'tanh'])
            dense_l2_reg = hp.Float(f'dense_l2_{i}', 0.0001, 0.001, sampling='LOG')
            dropout_rate = hp.Float(f'dropout_{i}', 0.2, 0.4, step=0.1)
            print(f"Dense layer {i}: units={dense_units}, activation={dense_activation}, l2_reg={dense_l2_reg}, dropout_rate={dropout_rate}")

            concat = Dense(units=dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2_reg))(concat)
            concat = Dropout(rate=dropout_rate)(concat)

        output = Dense(1)(concat)

        model = Model(inputs=[genex_input, smiles_input, drug_input], outputs=output)

        optimizer = hp.Choice('optimizer', values=['adam'])
        print(f"Optimizer: {optimizer}")

        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )

        return model
    except Exception as e:
        print(f"Error while building model: {e}")
        raise

# Implementing a custom callback to monitor epoch training time. If it excessive, we stop the
# training for this trial and move on.
class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_epoch_duration):
        super(TimeLimitCallback, self).__init__()
        self.max_epoch_duration = max_epoch_duration
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            if epoch_duration > self.max_epoch_duration:
                self.model.stop_training = True
                print(f"\nEpoch exceeded time limit of {self.max_epoch_duration} seconds. Stopping training for this trial.")
                self.epoch_start_time = None

max_epoch_duration = 600  # 10 minutes

# Initialize the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=25,
    executions_per_trial=1,
    directory='tuner_3',
    project_name='bayesian_optimization_large_range'
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Fewer epochs
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        TimeLimitCallback(max_epoch_duration=max_epoch_duration)
    ]
)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test data
test_loss, test_mse = best_model.evaluate(test_dataset)
print(f"Test MSE: {test_mse}")

# Tuner 5 (top 1000 genes)

# After running the optimal model found above on the top 1000 genes vs. all genes, I found I had better performance with the top 1000 genes.

# Here, I simplified the tuning process and performed one last tuning on the learning rate, dropout rate, and regularization parameter using the model architecture found at the previous tuning step.

def build_model(hp):
    # Hyperparameters to tune
    cnn_dropout_rate = hp.Float('cnn_dropout_rate', min_value=0.2, max_value=0.7, step=0.1)
    lstm_dropout_rate = hp.Float('lstm_dropout_rate', min_value=0.2, max_value=0.7, step=0.1)
    dense_dropout_rate = hp.Float('dense_dropout_rate', min_value=0.2, max_value=0.7, step=0.1)
    l2_reg_cnn = hp.Float('l2_reg_cnn', min_value=0.0001, max_value=0.01, sampling='log')
    l2_reg_lstm = hp.Float('l2_reg_lstm', min_value=0.0001, max_value=0.01, sampling='log')
    l2_reg_dense = hp.Float('l2_reg_dense', min_value=0.0001, max_value=0.01, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=0.00001, max_value=0.01, sampling='log')

    # CNN for gene expression data
    genex_input = Input(shape=genex_input_shape, name='genex_input')
    x = tf.keras.layers.Reshape((genex_input_shape[0], 1))(genex_input)

    filters = 4
    for _ in range(7):
        x = Conv1D(filters=filters, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg_cnn))(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(cnn_dropout_rate)(x)
        filters += 4

    x = Flatten()(x)
    genex_output = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg_dense))(x)

    # RNN for SMILES data
    smiles_input = Input(shape=smiles_input_shape, name='smiles_input')
    y = tf.keras.layers.Reshape((smiles_input_shape[0], 1))(smiles_input)
    y = LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg_lstm))(y)
    y = Dropout(lstm_dropout_rate)(y)
    y = LSTM(64, kernel_regularizer=l2(l2_reg_lstm))(y)
    y = Dropout(lstm_dropout_rate)(y)
    smiles_output = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg_dense))(y)

    # Concatenate outputs of CNN and RNN
    concat = Concatenate()([genex_output, smiles_output])

    # Add drug information input
    drug_input = Input(shape=drug_info_input_shape, name='drug_input')
    concat = Concatenate()([concat, drug_input])

    # Final fully connected network for prediction
    z = Dense(192, activation='relu', kernel_regularizer=l2(l2_reg_dense))(concat)
    z = Dropout(dense_dropout_rate)(z)
    output = Dense(1)(z)

    # Define the model
    model = Model(inputs=[genex_input, smiles_input, drug_input], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

# Initialize the tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=20,
    directory='tuner_1000_1',
    project_name='bayesian_optimization'
)

# Display search space summary
tuner.search_space_summary()

# Perform the search
tuner.search(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Print the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)

# Evaluate on test data
test_loss, test_mse = best_model.evaluate(test_dataset)
test_mse

# Model training and evaluation

# CNN for gene expression data
genex_input = Input(shape=genex_input_shape, name='genex_input')
x = tf.keras.layers.Reshape((genex_input_shape[0], 1))(genex_input)

base_filters = 4
for _ in range(7):
    x = Conv1D(filters=base_filters, kernel_size=3, activation='relu', kernel_regularizer=l2(0.0002))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.4)(x)
    base_filters += 4

x = Flatten()(x)
genex_output = Dense(128, activation='relu', kernel_regularizer=l2(0.0003))(x)

# RNN for SMILES data
smiles_input = Input(shape=smiles_input_shape, name='smiles_input')
y = tf.keras.layers.Reshape((smiles_input_shape[0], 1))(smiles_input)
y = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0009))(y)
y = Dropout(0.6)(y)
y = LSTM(64, kernel_regularizer=l2(0.0009))(y)
y = Dropout(0.6)(y)
smiles_output = Dense(128, activation='relu', kernel_regularizer=l2(0.0003))(y)

# Concatenate outputs of CNN and RNN
concat = Concatenate()([genex_output, smiles_output])

# Add drug information input
drug_input = Input(shape=drug_info_input_shape, name='drug_input')
concat = Concatenate()([concat, drug_input])

# Final fully connected network for prediction
z = Dense(192, activation='relu', kernel_regularizer=l2(0.0001))(concat)
z = Dropout(0.2)(z)
output = Dense(1)(z)

# Define the model
model = Model(inputs=[genex_input, smiles_input, drug_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

# Print the model summary
model.summary()

# Visual of the model
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=250,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath='cp_stat601_sasha_1000_2', save_weights_only=False)
    ]
)

# Visualize the training curves

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

plt.figure(figsize=(15, 5))

# Plot the loss curves
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the MSE curves
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_mse) + 1), train_mse, label='Training MSE')
plt.plot(range(1, len(val_mse) + 1), val_mse, label='Validation MSE')
plt.title('MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

test_loss, test_mse = model.evaluate(test_dataset)
test_mse
