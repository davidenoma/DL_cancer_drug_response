"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Biological data processing - filtering to LUAD cell lines
Author(s): Sasha Chernenkoff
Source: dl_cancer_drug_response.ipynb cells 3-13.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Data processing

# Biological processing (Sasha)
# We utilized two primary datasets: the GDSC (Genomics of Drug Sensitivity in Cancer) dataset for drug response data and gene expression data from the Cell Model Passports repository.

# The GDSC dataset provides detailed information on the sensitivity of various cancer cell lines (models) to a wide range of anti-cancer drugs. This includes dose-response data, where cell lines are exposed to different concentrations of drugs to determine the effect on cell viability. A key metric from this data is the IC50, which represents the concentration of a drug that is required to inhibit 50% of the cell population. Lower IC50 values indicate higher sensitivity of the cell line to the drug, which is important for identifying effective treatments for cancers.

# The gene expression data from Cell Model Passports consists of RNA-seq read counts, which measure the number of RNA transcripts produced by genes within the models. This data provides insights into the activity of various genes and helps to understand the molecular mechanisms driving cancer. By using the gene expression profiles, we can identify patterns in expression that correlate with drug sensitivity, enabling the prediction of IC50 values and thereby determining how effectively a drug can inhibit cancer cell growth.

# Data Filtering and Transformation
# To prepare the dataset for our analysis, we performed the following steps:
# 1. We loaded and filtered this dataset to include only the cell lines corresponding to lung adenocarcinoma (LUAD). Filtering by a single cancer type reduces the heterogeneity in the dataset, making it easier to identify patterns and draw conclusions specific to that cancer type. In the future, this can be expanded to other cancer types.

# 2. The unique drug names from the filtered GDSC dataset were extracted and saved to a text file. This will be used later when we incorporate the drug structures into our dataset.

# 3. The RNA-seq data was loaded and filtered to retain only the models present in the filtered GDSC dataset.

# 4. The final gene expression feature matrix was constructed with rows representing models and columns representing genes, containing the RNA-seq read counts.

# Imports
import pandas as pd

### Filtering by 'LUAD'

# Load data
print('Loading data...')
gdsc_df = pd.read_csv('../data/GDSC1_fitted_dose_response_27Oct23.csv')
expression_df = pd.read_csv('../data/rnaseq_read_count_20220624.csv', header=None)

# Filter GDSC1_fitted_dose_response_27Oct23.csv by TCGA_DESC = LUAD
filtered_gdsc_df = gdsc_df[gdsc_df['TCGA_DESC'] == 'LUAD']

# Get all unique SANGER_MODEL_ID values from the combined list
model_ids = filtered_gdsc_df['SANGER_MODEL_ID'].unique()

# Create drug response feature matrix
# Drop unneeded columns
filtered_gdsc_df.drop(columns=[
    'DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID',
    'CELL_LINE_NAME', 'TCGA_DESC', 'DRUG_ID', 'PUTATIVE_TARGET',
    'PATHWAY_NAME', 'COMPANY_ID', 'WEBRELEASE', 'RMSE', 'Z_SCORE',
    'AUC'
], inplace=True)

drugs = filtered_gdsc_df['DRUG_NAME'].unique()
# print(drugs)
# print(len(drugs))

# Save the list to a text file
with open('../data/unique_drugs.txt', 'w') as file:
    for drug in drugs:
        file.write(f"{drug}\n")

# Filter rnaseq_read_count_20220624.csv by model_ids (index by gene symbols)
# and convert to feature matrix: rows = models, cols = genes (rnaseq read counts)
names = ['model_id'] + expression_df.iloc[0, 2:].tolist()
expression_df = expression_df.iloc[5:, 1:]
expression_df.columns = names
filtered_columns = ['model_id'] + [col for col in expression_df.columns[2:] if col in model_ids]
filtered_expression_df = expression_df[filtered_columns]

# Save the filtered files
print('Saving filtered files...')
filtered_gdsc_df.to_csv('../data/filtered_drug_response.csv', index=False)
filtered_expression_df.to_csv('../data/filtered_rnaseq_read_count.csv', index=False)
