"""
Predicting patient drug responses using gene expression, drug SMILES and deep learning

Runs all src/ stages in notebook order.

Research code (Colab/Kaggle paths, cross-section globals); organized for reading
rather than guaranteed end-to-end execution.
"""
import runpy
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Pipeline stages, in notebook order.
STAGES = [
    "src/data_processing/biological_processing.py",
    "src/data_processing/gene_variance_filtering.py",
    "src/data_processing/drug_smiles_encoding.py",
    "src/data_processing/dataset_integration.py",
    "src/eda/exploratory_data_analysis.py",
    "src/models/models_investigation.py",
    "src/models/architecture_and_tuning.py",
    "src/models/multimodal_cnn_rnn.py",
    "src/models/cross_validation_tuning.py",
    "src/models/explainable_ai_shap.py",
    "src/models/fcnn_dimension_reduced.py",
]


def main():
    for rel_path in STAGES:
        path = os.path.join(HERE, rel_path)
        print(f"\n{'=' * 70}\n=== Running {rel_path}\n{'=' * 70}")
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
