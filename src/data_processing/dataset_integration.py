"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Flexible, memory-efficient dataset merging pipeline
Author(s): Mojtaba Kanani Sarcheshmeh
Source: dl_cancer_drug_response.ipynb cells 27-40.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Deep investigations/checking of the data, checking for biological and genetics consistency/correlations among features and merging for final data (Mojtaba)
# To ensure consistency in our approach to merging and analyzing the three different datasets, I developed a flexible and robust pipeline. This pipeline allows team members to specify their preferences for the number of genes and the type of drug chemical encodings to include in their analysis. By doing so, it caters to the varying needs and hypotheses of different team members without compromising the integrity of the process.
# By automating the merging and selection process, the pipeline minimizes human error and increases the reproducibility of the analyses. Each dataset generated follows the same rigorous methodology.

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import os

def memory_efficient_merge(df1, df2, left_on, right_on, how='inner', suffixes=('', '')):

  # Save df2 as csv
  df2.to_csv('/content/data/df2.csv', index=False)

  # Define the chunk size
  chunk_size = 1000  # Adjust based on your memory capacity

  # Create an empty DataFrame to store the merged results
  merged_df = pd.DataFrame()

  # Iterate over chunks of the first dataset
  for chunk in pd.read_csv('/content/data/df2.csv', chunksize=chunk_size):
      # Merge each chunk with the second dataset
      chunk_merged = pd.merge(df1, chunk, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)
      # Append the merged chunk to the result DataFrame
      merged_df = pd.concat([merged_df, chunk_merged], ignore_index=True)

  os.remove('/content/data/df2.csv')

  return merged_df

def encode_smile(smile, nBits):
  mol = Chem.MolFromSmiles(smile)
  fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
  return list(np.array([np.array(fp) for fp in fingerprints]))

def create_dataset(r_path = '/content/data/filtered_rnaseq_read_count.csv',
                   d_path = '/content/data/filtered_drug_response_all.csv',
                   s_path = '/content/data/drug_smiles.csv',
                   number_of_genes = -1,
                   drug_encoding_nBits = 1024):

  print("Reading Files ...")
  r = pd.read_csv(r_path)
  print("RNA seq Shape : ", r.shape)
  d = pd.read_csv(d_path)
  print("Drug Response Shape : ", d.shape)
  s = pd.read_csv(s_path)
  print("Drug SMILES Shape : ", s.shape)

  print("\nDropping NAs ...")
  r_na = r.index[r.iloc[:,1:].isna().sum(axis=1) > 0]
  print("Number of Rows with missing value in Response Data :",len(r_na))
  r = r.dropna().reset_index(drop=True)
  print("RNA seq Shape after dropping NAs: ", r.shape)

  if number_of_genes >= 0:
    print("\nReducing the number of Genes ...")
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    genes_index = r.select_dtypes(include=numerics).var(axis=1).sort_values(ascending=False)[:number_of_genes].index
    r = r.iloc[genes_index].reset_index(drop=True)
    print("RNA seq Shape after reducing the number of Genes: ", r.shape)

  print("\nTransposing ...")
  r = r.T.reset_index()
  r.columns = r.iloc[0]
  r = r.iloc[1:].reset_index(drop=True)
  print("RNA seq Shape after Transposing: ", r.shape)

  print("\nDropping 'Not Found' smiles ...")
  s_na = s[s['smile'] == "Not Found"].index
  print("Number of Rows with 'Not Found' in drug smiles data :",len(s_na))
  s = s[s['smile'] != "Not Found"].reset_index(drop=True)
  print("Drug SMILES Shape after dropping 'Not Found' smiles: ", s.shape)

  print('\nDropping drugs with unavailable SMILE ...')
  d_na = list(set(d['DRUG_NAME'].unique()) - set(s['drug'].unique()))
  print("Number of Rows with unavailable SMILE in drug response data :",len(d_na))
  d = d[d['DRUG_NAME'].isin(list(s['drug'].unique()))].reset_index(drop=True)
  print("Drug Response Shape after dropping drugs with unavailable SMILE: ", d.shape)

  print('\nEncoding SMILEs ...')
  s_enc = s['smile'].apply(lambda x: pd.Series(encode_smile(x, drug_encoding_nBits)))
  s_enc.columns = ['DrugChem_'+str(i+1) for i in range(drug_encoding_nBits)]
  s = pd.concat((s, s_enc), axis=1)
  s = s.drop('smile', axis=1)
  print("Drug SMILES Shape after encoding: ", s.shape)

  print('\nFiltering Unique Models in Drug Response Data ...')
  unique_models = list(set(r['model_id']))
  d = d[d['SANGER_MODEL_ID'].isin(unique_models)].reset_index(drop=True)
  print("Drug Response Shape after filtering unique models: ", d.shape)

  print('\nFiltering Unique combination of columns in Drug Response Data ...')
  d = d[~d.duplicated(subset=['SANGER_MODEL_ID', 'DRUG_NAME', 'MIN_CONC', 'MAX_CONC'])]
  print("Drug Response Shape after filtering unique combination of columns: ", d.shape)

  print('\nMerging Drug Response Data with Drug SMILES Data ...')
  merged = memory_efficient_merge(df1=d, df2=s, left_on='DRUG_NAME', right_on='drug', how='inner', suffixes=('', ''))
  print("Merged Data Shape : ", merged.shape)

  print('\nMerging Previously merged data with RNA seq Data ...')
  merged = memory_efficient_merge(df1=merged, df2=r, left_on='SANGER_MODEL_ID', right_on='model_id', how='inner', suffixes=('', ''))
  merged = merged.drop('drug', axis=1)
  print("Final Data Shape : ", merged.shape)

  file_name = 'final_'+ str(number_of_genes) if number_of_genes >= 0 else 'final_all'
  merged.to_csv('/content/data/'+str(file_name)+'.csv', index=False)
  print("Saved to : /content/data/"+str(file_name)+'.csv')

  return r, d, s, merged

r, d, s, merged = create_dataset(number_of_genes = 1000, 
                                 drug_encoding_nBits = 1024)

# !zip -r /content/data/final_1000.zip /content/data/final_1000.csv

r, d, s, merged = create_dataset(number_of_genes = -1,
                                 drug_encoding_nBits = 1024)

# !zip -r /content/data/final_all.zip /content/data/final_all.csv

# !zip -r /content/data/final.zip /content/data

# I have compiled two main datasets for our analysis:
# 1. Merged with top 1000 gene expressions:
#    - Number of Samples: 6474
#    - Number of Features: 1030
# 2. Merged with all genes:
#    - Number of Samples: 6474
#    - Number of Features: 38,293
