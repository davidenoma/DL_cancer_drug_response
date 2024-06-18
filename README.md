Title: Predicting patient drug responses using gene expresions, drugs SMILES and advanced deep learning techniques
Group members:
1. David Enoma
2. Sasha Chernenkoff
3. Ariel Ghislain Kemogne Kamdoum
4. Mojtaba Kanani Sarcheshmeh

Project Overview, architecture and objective
Background Story of the Project
Project Overview
The project aimed to create predictive models for drug sensitivity in various cancer cell lines. Utilizing extensive datasets, including Cell Model Passports, TCGA (The Cancer Genome Atlas), and the Genomics of Drug Sensitivity in Cancer (GDSC) database, the project aimed to integrate genetic and pharmacological data to predict drug efficacy. This endeavor combined machine learning, deep learning, and bioinformatics techniques to enhance personalized medicine approaches in oncology.

Data Sources
Cell Model Passports: This resource provided a comprehensive collection of cancer cell lines, their genetic profiles, and related metadata. The data included information from multiple cancers, allowing for a diverse and representative dataset.

TCGA: The Cancer Genome Atlas provided detailed genomic, epigenomic, transcriptomic, and proteomic data from numerous cancer types. This data was instrumental in understanding the genetic landscape of the cancer cell lines.

Genomics of Drug Sensitivity in Cancer (GDSC): This database contained drug response data, including IC50 values, which measure the effectiveness of drugs in inhibiting cancer cell growth. The data helped to correlate genetic profiles with drug sensitivity.

Data Preparation
RNAseq Data
RNA sequencing data was processed in chunks to manage memory usage efficiently. The project focused on calculating the variance of gene expressions across all samples to identify the most variable genes, which are likely the most informative for predictive modeling. The top 1000 genes with the highest variance were selected for further analysis.

Drug Response Data
Drug response data was merged with SMILES (Simplified Molecular Input Line Entry System) strings for drug molecules. These SMILES were converted into Morgan fingerprints using RDKit, which provided a standardized way to represent the chemical structure of drugs. The fingerprints were then scaled for consistency and merged with the RNAseq data to form a comprehensive dataset.

Model Development
Random Forest Regressor: A Random Forest model was trained to predict the IC50 values of drugs based on the combined genetic and pharmacological data. This ensemble method leveraged the power of multiple decision trees to improve prediction accuracy.

Multiple Linear Regression (MLR): A simple linear regression model was also developed to provide a baseline for comparison with more complex models.

Optimized Multi-Layer Perceptron (MLP): Using Keras Tuner, a hyperparameter search was conducted to optimize a deep learning model. This MLP model aimed to capture non-linear relationships in the data, potentially offering better predictive performance than traditional methods.

Convolutional Neural Network (CNN):

Recurrent Neural Network (RNN):

eXplanaible AI on Deep models via SHAP with PCA and SHAP with autoencoder: Using PCA (for hanlding linearity) and autoencoder (for handling nonlinearity) we reduce the dimension of the data to observe the contribution and importance of particular features on the model's performance and predictions

Hyperparameter Tuning: Using several deep learning hyperparameter tuning techniques, we impl

Results and Evaluation
The models were evaluated using metrics like Mean Squared Error (MSE) and R² score to measure their predictive accuracy. The Random Forest model and the optimized MLP model showed promising results, demonstrating the potential of integrating genetic and drug response data for predicting drug sensitivity.

Future Work
The project lays the groundwork for more advanced models and larger-scale studies. Future directions include:

Incorporating more complex features from genetic data, such as epigenetic modifications and protein expression levels.
Exploring other machine learning algorithms and deep learning architectures.
Applying transfer learning techniques to leverage pre-trained models on similar datasets.
Conducting prospective validation studies to test the models on new data.
Conclusion
This project successfully combined extensive datasets and advanced machine learning techniques to predict drug sensitivity in cancer cell lines. The integration of genetic profiles with pharmacological data provides a robust framework for personalized medicine, offering a pathway towards more effective and tailored cancer treatments.
