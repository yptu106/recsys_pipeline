import torch
import torch.nn as nn

class MLPRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a relevance score
        )

    def forward(self, user_emb, item_emb):
        interaction = user_emb * item_emb
        x = torch.cat([user_emb, item_emb, interaction], dim=-1)  # Concatenate user and item embeddings (shape: 3 * emb_dim)

        return self.mlp(x).squeeze(-1)  # shape: [batch_size]
