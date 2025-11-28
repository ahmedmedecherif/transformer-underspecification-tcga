import torch
import pandas as pd
import numpy as np
from model import GeneTransformer

"""
Evaluate OOD stability using GTEx expression data.
GTEx is unlabeled for cancer; predictions are used ONLY as a stability measure,
not a biological accuracy metric.
"""

# -----------------------------
# Load GTEx data
# -----------------------------
gtex = pd.read_csv("gtex_expression.csv", index_col=0)

# Same preprocessing as TCGA
gtex = np.log1p(gtex)
gtex = (gtex - gtex.mean()) / (gtex.std() + 1e-6)

# Load selected genes
selected = pd.read_csv("selected_genes.csv", header=None)[0].values
gtex = gtex[selected]

gtex_tensor = torch.tensor(gtex.values, dtype=torch.float32)

# -----------------------------
# Load trained Transformer models
# -----------------------------
num_classes = 32  # adjust if needed

models = {}
for seed in [1, 2, 3, 4, 5]:
    model = GeneTransformer(num_genes=gtex_tensor.shape[1], num_classes=num_classes)
    model.load_state_dict(torch.load(f"model_seed{seed}.pt", map_location="cpu"))
    model.eval()
    models[seed] = model

# -----------------------------
# OOD Predictions (proxy for stability)
# -----------------------------
stability = {}

for seed, model in models.items():
    with torch.no_grad():
        pred = model(gtex_tensor)
        pred_classes = pred.argmax(dim=1).numpy()
        stability[f"seed_{seed}"] = pred_classes

# Save predictions for analysis
stability_df = pd.DataFrame(stability)
stability_df.to_csv("gtex_predictions_across_seeds.csv")

print("OOD predictions saved to gtex_predictions_across_seeds.csv.")
