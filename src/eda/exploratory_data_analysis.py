"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Exploratory data analysis, pre-processing and cleaning
Author(s): Ariel Ghislain Kemogne Kamdoum
Source: dl_cancer_drug_response.ipynb cells 43-58.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Exploration Data Analysis, pre-processing and cleaning (Ariel)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import pandas as pd


data = pd.read_csv('/content/sample_data/final_drug_modelid_minmaxconc_unique.csv')
data

# Display basic statistics and first few rows of the dataset
print(data.info())
print(data.describe(include='all'))
print(data.head())

# Check for missing values
missing_values = data.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
print(missing_values)

# Distribution of the target variable (LN_IC50)
plt.figure(figsize=(10, 6))
sns.histplot(data['LN_IC50'], bins=50, kde=True)
plt.title('Distribution of LN_IC50')
plt.xlabel('LN_IC50')
plt.ylabel('Frequency')
plt.show()

# Drop columns that contain string values
data = data.select_dtypes(exclude=['object'])

# Encode categorical variables
label_encoder = LabelEncoder()
data['model_id'] = label_encoder.fit_transform(data['model_id'])
data['drug'] = label_encoder.fit_transform(data['drug'])

# Correlation heatmap of top 20 most correlated features with LN_IC50
correlation_matrix = data.corr()
top_corr_features = correlation_matrix['LN_IC50'].abs().sort_values(ascending=False).index[1:21]
plt.figure(figsize=(14, 10))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Top 20 Most Correlated Features with LN_IC50')
plt.show()

# Visualizing the relationship between a few selected gene expressions and LN_IC50
selected_genes = ['MT-RNR2', 'FN1', 'MT-CO1', 'MT-ND4', 'EEF1A1']
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
axes = axes.flatten()
for i, gene in enumerate(selected_genes):
    sns.scatterplot(ax=axes[i], x=gene, y='LN_IC50', data=data)
    axes[i].set_title(f'{gene} vs LN_IC50')
axes[-1].axis('off')
plt.tight_layout()
plt.show()

# Load the dataset
file_path = '/content/sample_data/final_drug_modelid_minmaxconc_unique.csv'
data = pd.read_csv(file_path)

# Encode the 'model_id' column only
label_encoder = LabelEncoder()
data['model_id'] = label_encoder.fit_transform(data['model_id'])

# Plot the distribution of the top 20 most frequent drugs with their names
plt.figure(figsize=(14, 7))
top_20_drugs = data['drug'].value_counts().index[:20]
top_20_drug_counts = data['drug'].value_counts().values[:20]
sns.barplot(x=top_20_drugs, y=top_20_drug_counts)
plt.title('Top 20 Most Frequent Drugs')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
file_path = '/content/sample_data/final_drug_modelid_minmaxconc_unique.csv'
data = pd.read_csv(file_path)

# Encode the 'model_id' column only
label_encoder = LabelEncoder()
data['model_id'] = label_encoder.fit_transform(data['model_id'])

# Drop non-numeric columns for PCA
gene_data = data.drop(columns=['model_id', 'drug', 'MIN_CONC', 'MAX_CONC', 'LN_IC50'])

# Remove columns with all zeros
gene_data = gene_data.loc[:, (gene_data != 0).any(axis=0)]

# Ensure all columns are numeric
gene_data = gene_data.select_dtypes(include=[np.number])

# Handle missing values by filling with the mean of each column
gene_data = gene_data.fillna(gene_data.mean())

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(gene_data)

# Create a DataFrame with PCA results and the drug names
pca_df = pd.DataFrame(data={'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'Drug': data['drug']})

# Create a scatter plot with annotations
plt.figure(figsize=(14, 10))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Drug', palette='viridis', legend='full')

# Annotate each point with the drug name
for i in range(pca_df.shape[0]):
    plt.text(pca_df['PC1'][i], pca_df['PC2'][i], str(pca_df['Drug'][i]), fontsize=9, ha='right')

plt.title('PCA of Gene Expressions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Pairplot of top 5 most correlated features with LN_IC50
top_5_corr_features = top_corr_features[:5].tolist() + ['LN_IC50']
sns.pairplot(data[top_5_corr_features])
plt.suptitle('Pairplot of Top 5 Features Most Correlated with LN_IC50', y=1.02)
plt.show()

#Feature Importance Using Random Forest
# We use a Random Forest model to determine the feature importance for predicting LN_IC50.

from sklearn.ensemble import RandomForestRegressor
importances = []

# Fit a Random Forest model
X = data_cleaned.drop(columns=['LN_IC50'])
y = data_cleaned['LN_IC50']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for the feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

# Plot the top 20 feature importances
plt.figure(figsize=(14, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 20 Feature Importances for Predicting LN_IC50')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#Clustering Analysis
#Perform clustering on the gene expression data to identify possible subgroups.

from sklearn.cluster import KMeans

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(gene_data)

# Add cluster information to the PCA DataFrame
pca_df['Cluster'] = clusters

# Plot the PCA with cluster information
plt.figure(figsize=(14, 10))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', legend='full')
plt.title('PCA of Gene Expressions with KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Heatmap of Gene Expression Data
#Visualize the heatmap of the gene expression data to identify patterns.

# Randomly sample 100 genes for better visualization
sampled_genes = gene_data.sample(n=100, axis=1)

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(sampled_genes, cmap='viridis')
plt.title('Heatmap of Randomly Sampled Gene Expressions')
plt.xlabel('Gene')
plt.ylabel('Sample')
plt.show()

#Drug Response Analysis
#Analyze how different drugs affect the LN_IC50.

# Calculate mean LN_IC50 for each drug
drug_response = data.groupby('drug')['LN_IC50'].mean().sort_values()

# Plot the drug response
plt.figure(figsize=(14, 7))
sns.barplot(x=drug_response.index[:20], y=drug_response.values[:20])
plt.title('Top 20 Drugs with Lowest Mean LN_IC50')
plt.xlabel('Drug')
plt.ylabel('Mean LN_IC50')
plt.xticks(rotation=90)
plt.show()
