# Source scripts

`dl_cancer_drug_response.ipynb` reorganized into per-section scripts. Code and
comments verbatim; notebook markdown retained as comments. Notebook shell
commands (`!pip`, `!zip`) are commented out; pasted training logs are omitted.

Run in notebook order with `python main.py`, or run a single script directly.

Research code (Colab/Kaggle paths, cross-section globals); organized for reading
rather than guaranteed end-to-end execution.

## Layout (pipeline order)

| Script | Section | Author |
| --- | --- | --- |
| `data_processing/biological_processing.py` | Filtering to LUAD cell lines | Sasha |
| `data_processing/gene_variance_filtering.py` | Top-1000 gene selection by variance | David |
| `data_processing/drug_smiles_encoding.py` | SMILES to Morgan fingerprints & integration | Ariel |
| `data_processing/dataset_integration.py` | Flexible dataset merge pipeline | Mojtaba |
| `eda/exploratory_data_analysis.py` | EDA, pre-processing, cleaning | Ariel |
| `models/models_investigation.py` | Baselines, NN, CNN, deep NN | Ariel |
| `models/architecture_and_tuning.py` | RF/MLR/MLP/CNN tuning | David |
| `models/multimodal_cnn_rnn.py` | Multimodal CNN + RNN | Sasha |
| `models/cross_validation_tuning.py` | Cross-validation & tuning | Ariel |
| `models/explainable_ai_shap.py` | SHAP (PCA & autoencoder) | Ariel |
| `models/fcnn_dimension_reduced.py` | PCA + fully-connected model | Mojtaba |

## Project overview

# Title: Predicting patient drug responses using gene expresions, drugs SMILES and advanced deep learning techniques

## Authors:David Enoma, Sasha Chernenkoff, Ariel Ghislain Kemogne Kamdoum, Mojtaba Kanani Sarcheshmeh

### Background Story of the Project

### Project Overview

The project aimed to create predictive models for drug sensitivity in various cancer cell lines. Utilizing extensive datasets, including Cell Model Passports, TCGA (The Cancer Genome Atlas), and the Genomics of Drug Sensitivity in Cancer (GDSC) database, the project aimed to integrate genetic and pharmacological data to predict drug efficacy based on IC50 Values (the concentration of a drug that causes Effectivity by 50% on cell lines). This endeavor combined machine learning, deep learning, and bioinformatics techniques to enhance personalized medicine approaches in oncology.

### Data Sources

1. **Cell Model Passports**: This resource provided a comprehensive collection of cancer cell lines, their genetic profiles, and related metadata. The data included information from multiple cancers, allowing for a diverse and representative dataset.
   
2. **TCGA**: The Cancer Genome Atlas provided detailed genomic, epigenomic, transcriptomic, and proteomic data from numerous cancer types. This data was instrumental in understanding the genetic landscape of the cancer cell lines.
   
3. **Genomics of Drug Sensitivity in Cancer (GDSC)**: This database contained drug response data, including IC50 values, which measure the effectiveness of drugs in inhibiting cancer cell growth. The data helped to correlate genetic profiles with drug sensitivity.

### Data Preparation

#### RNAseq Data

RNA sequencing data was processed in chunks to manage memory usage efficiently. The project focused on calculating the variance of gene expressions across all samples to identify the most variable genes, which are likely the most informative for predictive modeling. The top 1000 genes with the highest variance were selected for further analysis.

#### Drug Response Data

Drug response data was merged with SMILES (Simplified Molecular Input Line Entry System) strings for drug molecules. These SMILES were converted into Morgan fingerprints using RDKit, which provided a standardized way to represent the chemical structure of drugs. The fingerprints were then scaled for consistency and merged with the RNAseq data to form a comprehensive dataset.

### Model Development

1. **Random Forest Regressor**: A Random Forest model was trained to predict the IC50 values of drugs based on the combined genetic and pharmacological data. This ensemble method leveraged the power of multiple decision trees to improve prediction accuracy.
   
2. **Multiple Linear Regression (MLR)**: A simple linear regression model was also developed to provide a baseline for comparison with more complex models.
   
3. **Optimized Multi-Layer Perceptron (MLP)**: Using Keras Tuner, a hyperparameter search was conducted to optimize a deep learning model. This MLP model aimed to capture non-linear relationships in the data, potentially offering better predictive performance than traditional methods.

4. **Convolutional Neural Network (CNN)**:

5. **Recurrent Neural Network (RNN)**:

6. **eXplanaible AI on Deep models via SHAP with PCA and SHAP with autoencoder**: Using PCA (for hanlding linearity) and autoencoder (for handling nonlinearity) we reduce the dimension of the data to observe the contribution and importance of particular features on the model's performance and predictions

6. **Hyperparameter Tuning:** Using several deep learning hyperparameter tuning techniques, Grid Search and Bayesian optimization.

### Results and Evaluation

The models were evaluated using Mean Squared Error (MSE) and R² score metrics to measure their predictive accuracy. The optimized CNN model achieved the best performance with an MSE of 1.1598 and an R² score of 0.8316, followed by the Random Forest model with an MSE of 1.2844 and an R² score of 0.8135, and the optimized MLP model with an MSE of 1.3339 and an R² score of 0.8063. These results highlight the potential of integrating genetic and drug response data for accurately predicting drug sensitivity.

### Future Work

The project lays the groundwork for more advanced models and larger-scale studies. Future directions include:

- Incorporating more complex features from genetic data, such as epigenetic modifications and protein expression levels.
- Exploring other machine learning algorithms and deep learning architectures.
- Applying transfer learning techniques to leverage pre-trained models on similar datasets.
- Conducting prospective validation studies to test the models on new data.

### Conclusion

This project successfully combined extensive datasets and advanced machine learning techniques to predict drug sensitivity in cancer cell lines. The integration of genetic profiles with pharmacological data provides a robust framework for personalized medicine, offering a pathway towards more effective and tailored cancer treatments.

## Summary

# Summary

### Performance comparison

Of all the models compared, David's CNN model was the best performing with an MSE of 1.1598.
The study evaluated various models for predicting drug sensitivity using Mean Squared Error (MSE) and R² score metrics. The Random Forest model and an optimized Multi-Layer Perceptron (MLP) showed promising results, indicating the value of combining genetic and drug response data.

Key Results:
- **Optimized CNN**: MSE = 1.1598, R² = 0.8316, Mean Absolute Error = 0.8099
- **Optimized MLP**: MSE = 1.3339, R² = 0.8063
- **Random Forest**: MSE = 1.2844, R² = 0.8135
- **Multiple Linear Regression**: MSE = 1.3561, R² = 0.8031

The optimized CNN, with specific convolutional and dense layers detailed in the `build_large_cnn` function, performed the best in terms of R² score, followed by the Random Forest and optimized MLP models.

### Future work

This model can be extended to include other cancers.

# Acknowledgment
**We are deeply grateful to Dr. Zhang for their invaluable guidance, insightful feedback, and unwavering support throughout this project.**
