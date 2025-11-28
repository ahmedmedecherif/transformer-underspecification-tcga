import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

"""
Preprocessing pipeline for TCGA gene expression data.
This script performs:
- log1p transform
- z-score normalization using train split statistics
- ANOVA F-test to select top N genes
- Saves preprocessed expression matrices and labels
"""

# ---- Load raw data ----
tcga_expr = pd.read_csv("tcga_expression_raw.csv", index_col=0)
tcga_labels = pd.read_csv("tcga_labels.csv", index_col=0)["cancer_type"]

# ---- Log transform ----
tcga_expr = np.log1p(tcga_expr)

# ---- Train/test split (stratified) ----
X_train, X_test, y_train, y_test = train_test_split(
    tcga_expr, tcga_labels, test_size=0.2, random_state=42, stratify=tcga_labels
)

# ---- Z-scoring using train statistics only ----
mean = X_train.mean()
std = X_train.std() + 1e-6
X_train_z = (X_train - mean) / std
X_test_z = (X_test - mean) / std

# ---- ANOVA F-test (top 1500 genes) ----
f_scores, _ = f_classif(X_train_z, y_train)
top_genes = np.argsort(f_scores)[-1500:]
selected_cols = X_train_z.columns[top_genes]

X_train_sel = X_train_z[selected_cols]
X_test_sel = X_test_z[selected_cols]

# ---- Save outputs ----
X_train_sel.to_csv("X_train_processed.csv")
X_test_sel.to_csv("X_test_processed.csv")
y_train.to_csv("y_train.csv")
y_test.to_csv("y_test.csv")
pd.Series(selected_cols).to_csv("selected_genes.csv", index=False)

print("Preprocessing completed. Files saved:")
print("X_train_processed.csv, X_test_processed.csv, y_train.csv, y_test.csv, selected_genes.csv")
