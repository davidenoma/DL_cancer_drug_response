"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Section: Drug chemistry: encoding SMILES as Morgan fingerprints and integrating them
Author(s): Ariel Ghislain Kemogne Kamdoum
Source: dl_cancer_drug_response.ipynb cells 16-23.

Research code (Colab/Kaggle paths, cross-section globals); organized for
reading rather than guaranteed end-to-end execution.
"""


# Drugs Chemistry and encoding of Drugs SMILES (Ariel)

# Biological Understanding of Molecular Fingerprint Encoding
# Introduction
# In the realm of computational biology and chemoinformatics, the task of representing chemical compounds in a format suitable for computational analysis is paramount. One of the most widely used representations of chemical compounds is the Simplified Molecular Input Line Entry System (SMILES). However, for advanced computational tasks, such as those involving machine learning and deep learning, SMILES strings need to be converted into numerical representations that preserve the chemical and structural properties of the molecules. This section work discusses the process of encoding SMILES strings into molecular fingerprints using Morgan fingerprints, a powerful method for capturing the intricate details of molecular structures.

# SMILES to RDKit Molecule Conversion
# SMILES strings are a linear textual representation of molecular structures, where atoms are represented by their chemical symbols and bonds by specific characters. While SMILES is highly compact and human-readable, it does not directly lend itself to numerical analysis. The first step in transforming SMILES strings into a usable format is to convert them into molecular objects using a cheminformatics library like RDKit. RDKit is a widely-used toolkit that provides tools for cheminformatics, including functionalities for parsing SMILES strings into molecular objects (Mol objects).

# Morgan Fingerprints: Concept and Generation
# Morgan fingerprints, also known as circular fingerprints, are a type of molecular fingerprint that captures the presence of substructures within a molecule. They are generated using a circular substructure pattern where each atom and its neighboring atoms up to a certain radius are considered. This method provides a unique and comprehensive representation of the molecule's topology and chemical environment.

# To generate Morgan fingerprints, the following steps are undertaken:
# Initialization: Each atom in the molecule is assigned an initial identifier based on its atomic number and other properties.

# Neighborhood Iteration: For a given radius, the algorithm iteratively updates each atom's identifier based on the identifiers of its neighboring atoms. This process continues for the specified number of iterations (equivalent to the radius).

# Bit Vector Encoding: The final identifiers are hashed into a fixed-length bit vector, typically 1024 bits long, where each bit represents the presence or absence of a particular substructure.

# This bit vector, or fingerprint, effectively captures the structural information of the molecule in a binary format that is amenable to machine learning algorithms.

# Conversion to Numerical Array
# Once the Morgan fingerprints are generated for each molecule, they are converted into a numerical array. Each molecule is represented as a vector of fixed length, where each position in the vector corresponds to a specific substructure pattern. The presence of a pattern is indicated by a bit value of 1, while its absence is indicated by a bit value of 0. This transformation results in a highly compact and information-rich numerical representation of the molecule.

# Integration with Drug Data
# In a practical scenario, the encoded fingerprints are integrated with additional drug-related data. For instance, each drug's SMILES string is associated with its name or identifier. The numerical fingerprints are concatenated with this metadata, creating a comprehensive dataset that includes both the chemical structure and relevant identifiers for each drug.

# Applications in Computational Biology
# The numerical encoding of molecular structures using Morgan fingerprints has several applications in computational biology and drug discovery:
# Drug Similarity and Clustering: The fingerprints can be used to compute similarity between drugs, enabling the clustering of similar compounds and the identification of potential analogs.

# Machine Learning Models: Encoded fingerprints serve as input features for machine learning models, facilitating tasks such as drug activity prediction, toxicity estimation, and virtual screening.

# Feature Extraction and Analysis: The bit vectors can be analyzed to extract meaningful features that correlate with biological activity, aiding in the understanding of structure-activity relationships.

# Conclusion
# The process of converting SMILES strings to Morgan fingerprints and subsequently to numerical arrays is a critical step in the computational analysis of chemical compounds. This encoding captures the essential structural characteristics of molecules, enabling their use in a wide range of computational biology applications. By transforming textual SMILES representations into compact and informative numerical vectors, researchers can leverage advanced machine learning techniques to accelerate drug discovery and development.

# Summary
# SMILES Strings: Linear textual representation of molecular structures.

# RDKit Molecules: Conversion of SMILES to molecular objects.

# Morgan Fingerprints: Generation of circular fingerprints capturing substructures within molecules.

# Numerical Encoding: Transformation of fingerprints into binary vectors.

# Applications: Drug similarity, machine learning models, feature extraction, and analysis.

# This detailed encoding process ensures that the complex structural information of molecules is preserved and made accessible for computational analysis, driving forward innovations in drug discovery and other areas of computational biology.

# !pip install rdkit

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the CSV file
file_path = '/content/sample_data/drugs_filtered.csv'
drugs_df = pd.read_csv(file_path)

# Convert SMILES to RDKit Mol objects
all_smiles = drugs_df['smiles']
mols = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]

# Generate Morgan Fingerprints
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]

# Convert fingerprints to numpy array
fingerprints_array = np.array([np.array(fp) for fp in fingerprints])

# Add fingerprints to DataFrame
fingerprints_df = pd.DataFrame(fingerprints_array)
encoded_drugs_df = pd.concat([drugs_df['drugs'], fingerprints_df], axis=1)

# Save the final DataFrame to a new CSV file
output_file_path = '/content/sample_data/Encoded_Drugs.csv'
encoded_drugs_df.to_csv(output_file_path, index=False)

print(f"Encoded SMILES saved to: {output_file_path}")
encoded_drugs_df

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the CSV file
file_path = '/content/sample_data/drug_smiles.csv'
drugs_df = pd.read_csv(file_path)

# Function to validate and convert SMILES to RDKit Mol objects
def smiles_to_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

# Convert SMILES to RDKit Mol objects, filtering out invalid ones
all_smiles = drugs_df['smile']
mols = [smiles_to_mol(smiles) for smiles in all_smiles]
valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
valid_mols = [mol for mol in mols if mol is not None]
valid_drugs = drugs_df.iloc[valid_indices]

# Generate Morgan Fingerprints for valid molecules
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in valid_mols]

# Convert fingerprints to numpy array
fingerprints_array = np.array([np.array(fp) for fp in fingerprints])

# Add fingerprints to DataFrame
fingerprints_df = pd.DataFrame(fingerprints_array)
encoded_drugs_df = pd.concat([valid_drugs.reset_index(drop=True), fingerprints_df], axis=1)

# Save the final DataFrame to a new CSV file
output_file_path = '/content/sample_data/encoded_drugs.csv'
encoded_drugs_df.to_csv(output_file_path, index=False)

print(f"Encoded SMILES saved to: {output_file_path}")
encoded_drugs_df

encoded_drugs_df

# Integrating the drugs encoded smiles in the data (Ariel)

import pandas as pd

# Load the CSV files
reduced_rnaseq_read_count_path = '/content/sample_data/reduced_rnaseq_read_count_all.csv'
filtered_drug_response_path = '/content/sample_data/filtered_drug_response_all.csv'
encoded_drugs_path = '/content/sample_data/encoded_drugs (1).csv'

# Read the dataframes
reduced_rnaseq_df = pd.read_csv(reduced_rnaseq_read_count_path)
filtered_drug_response_df = pd.read_csv(filtered_drug_response_path)
encoded_drugs_df = pd.read_csv(encoded_drugs_path)

# Rename columns for consistency
filtered_drug_response_df.rename(columns={'SANGER_MODEL_ID': 'model_id', 'DRUG_NAME': 'drug'}, inplace=True)

# Print column names to identify correct columns
print("Reduced RNASeq Columns:", reduced_rnaseq_df.columns)
print("Filtered Drug Response Columns:", filtered_drug_response_df.columns)
print("Encoded Drugs Columns:", encoded_drugs_df.columns)

# Merge the drug response with encoded drugs based on 'drug' column
merged_drug_df = pd.merge(filtered_drug_response_df, encoded_drugs_df, on='drug')

# Merge the result with reduced RNASeq data based on 'model_id'
final_merged_df = pd.merge(reduced_rnaseq_df, merged_drug_df, on='model_id')

# Save the final merged dataframe to a new CSV file
output_merged_file_path = '/content/sample_data/final_merged_data.csv'
final_merged_df.to_csv(output_merged_file_path, index=False)

print(f"Merged data saved to: {output_merged_file_path}")
