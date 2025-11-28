import torch
import torch.nn as nn

class GeneTransformer(nn.Module):
    """
    Transformer model for gene expression classification.
    Input shape: (batch_size, num_genes)
    The model embeds each gene as a token and applies a Transformer encoder.
    """

    def __init__(self, num_genes=1500, num_classes=32, d_model=256, nhead=4, num_layers=4):
        super().__init__()

        # Embed each gene as a token
        self.embedding = nn.Linear(1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x shape: (batch, genes)
        """
        x = x.unsqueeze(-1)          # -> (batch, genes, 1)
        x = self.embedding(x)        # -> (batch, genes, d_model)
        x = x.permute(1, 0, 2)       # -> (genes, batch, d_model)
        h = self.encoder(x)          # -> (genes, batch, d_model)
        h_avg = h.mean(dim=0)        # -> (batch, d_model)
        out = self.fc(h_avg)         # -> (batch, num_classes)
        return out
