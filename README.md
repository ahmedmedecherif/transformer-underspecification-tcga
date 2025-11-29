# Transformer Underspecification in TCGA Cancer Classification  
Code for the manuscript:

**"Assessing Underspecification in Transformer-Based Cancer Classification Models Using Gene Expression Data"**  
*(Submitted to BMC Bioinformatics, 2025)*

---

## 🔍 Overview

This repository contains the full reproducible code used in our study analyzing **underspecification** in Transformer-based models trained on TCGA gene expression data.

The study evaluates:

- Stability of in-distribution (ID) performance across multiple random seeds  
- Divergence in model behavior under external **out-of-distribution (OOD)** shift (GTEx)  
- Consistency of feature attribution using **Integrated Gradients (IG)**  
- Variability in learned gene–gene interaction patterns extracted from **multi-head attention maps**  
- Sensitivity of Transformer decision boundaries to seed initialization and small perturbations  

These analyses support the manuscript’s central finding:  
**Transformers achieve strong ID accuracy yet encode unstable decision boundaries, revealing underspecification in transcriptomic classification tasks.**

---

## 📁 Repository Structure
## Preprint / PDF
A PDF version of the manuscript is available at:
https://github.com/ahmedmedecherif/transformer-underspecification-tcga/blob/main/manuscript.pdf

Please cite the repository as:
Ech-Cherif A, et al. (2025).
Transformer Underspecification in TCGA Cancer Classification.
GitHub: https://github.com/ahmedmedecherif/transformer-underspecification-tcga


