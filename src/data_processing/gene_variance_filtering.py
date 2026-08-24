"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Downsizing gene expression by variance (top 1000 genes)
Author(s): David Enoma
Source: dl_cancer_drug_response.ipynb cells 14-15.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Downsize/Filtering gene expression based on variants (David)

import pandas as pd
import numpy as np
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
