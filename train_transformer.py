import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import GeneTransformer

"""
Train the Transformer classifier across multiple seeds
to assess underspecification and model stability.
"""

# -----------------------------
# Load processed data
# -----------------------------
X_train = pd.read_csv("X_train_processed.csv", index_col=0).values
y_train = pd.read_csv("y_train.csv", index_col=0).values.squeeze()

X_test = pd.read_csv("X_test_processed.csv", index_col=0).values
y_test = pd.read_csv("y_test.csv", index_col=0).values.squeeze()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=32)

# -----------------------------
# Training function
# -----------------------------
def train_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_genes   = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = GeneTransformer(
        num_genes=num_genes,
        num_classes=num_classes,
        d_model=256,
        nhead=4,
        num_layers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- Training loop ----
    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            correct += (pred.argmax(dim=1) == yb).sum().item()
            total   += yb.size(0)

    return correct / total


# -----------------------------
# Multi-seed training
# -----------------------------
results = {}
for seed in [1, 2, 3, 4, 5]:
    print(f"\nTraining seed {seed}...")
    model = train_model(seed)
    acc   = evaluate(model)

    results[f"seed_{seed}"] = acc
    torch.save(model.state_dict(), f"model_seed{seed}.pt")

pd.Series(results).to_csv("seed_results.csv")
print("\nFinished training. Results saved to seed_results.csv.")
print(results)
