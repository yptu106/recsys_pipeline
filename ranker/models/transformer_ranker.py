import torch
import torch.nn as nn

class TransformerRanker(nn.Module):
    def __init__(self, input_dim, d_model=256):
        super().__init__()
        self.embedding_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, user_emb, item_emb):
        # Project to shared space
        u = self.embedding_proj(user_emb)[:, None, :]  # (B, 1, D)
        i = self.embedding_proj(item_emb)[:, None, :]  # (B, 1, D)

        # Forming a two-token sequence
        # - Token 0 = the user representation
        # - Token 1 = the item representation
        x = torch.cat([u, i], dim=1)  # (B, 2, D)

        x = self.transformer(x)      # (B, 2, D)

        # MLP on the first token (user representation)
        out = self.mlp(x[:, 0, :])   # use the first token ([CLS]) as the summary representation
        return out.squeeze()
