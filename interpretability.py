import torch
import pandas as pd
from captum.attr import IntegratedGradients
from model import GeneTransformer
import numpy as np

"""
Compute Integrated Gradients and attention statistics
to analyze model interpretability across seeds.
This supports Figures 3 and 4 of the manuscript.
"""

# -----------------------------
# Load data (test-set for IG)
# -----------------------------
X_test = pd.read_csv("X_test_processed.csv", index_col=0).values
X_test = torch.tensor(X_test, dtype=torch.float32)

# -----------------------------
# Load trained model (choose a seed)
# -----------------------------
model = GeneTransformer(
    num_genes=X_test.shape[1],
    num_classes=32,
    d_model=256,
    nhead=4,
    num_layers=4
)
model.load_state_dict(torch.load("model_seed1.pt", map_location="cpu"))
model.eval()

# -----------------------------
# Integrated Gradients
# -----------------------------
ig = IntegratedGradients(model)

# Pick a representative instance
x_sample = X_test[0].unsqueeze(0)

# Compute IG attribution scores
attributions = ig.attribute(x_sample, target=None)
attributions = attributions.squeeze().detach().numpy()

# Save IG scores
pd.Series(attributions).to_csv("ig_scores_seed1.csv")

print("IG scores saved → ig_scores_seed1.csv")


# -----------------------------
# Attention extraction utility
# -----------------------------
def get_attention_maps(model, x):
    """
    Hook into the Transformer encoder to extract attention weights.
    """

    attention_maps = []

    def hook(module, input, output):
        # The multi-head attention submodule stores weights in output[1]
        if isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1].detach().numpy())

    # Register hook for all MultiheadAttention modules
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.register_forward_hook(hook)

    # Forward pass
    _ = model(x.unsqueeze(0))

    return attention_maps


# -----------------------------
# Compute attention maps
# -----------------------------
attn_maps = get_attention_maps(model, X_test[0])

# Save the raw attention maps
np.save("attention_maps_seed1.npy", attn_maps)

print("Attention maps saved → attention_maps_seed1.npy")
