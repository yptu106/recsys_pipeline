import torch
import torch.nn as nn

class MLPRanker(nn.Module):
    def __init__(self, input_dim, d_model=256):
        super().__init__()
        self.user_proj = nn.Linear(input_dim, d_model)
        self.item_proj = nn.Linear(input_dim, d_model)

        # final input: [user_proj, item_proj, user_proj * item_proj] => 3 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a relevance score
            # TODO: add l2-normalization
        )


    def forward(self, user_emb, item_emb):
        user_emb = self.user_proj(user_emb) # (B, D)
        item_emb = self.item_proj(item_emb) # (B, D)
        interaction = user_emb * item_emb   # Element-wise product (B, D)

        x = torch.cat([user_emb, item_emb, interaction], dim=-1) # (B, 3D)
        score = self.mlp(x).squeeze(-1)  # (B,)

        return score 
