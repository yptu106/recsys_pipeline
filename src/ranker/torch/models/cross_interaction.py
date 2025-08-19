import torch
import torch.nn as nn

class CrossInteractionRanker(nn.Module):
    def __init__(self, input_dim, d_model=256):
        super().__init__()
        self.user_proj = nn.Linear(input_dim, d_model)
        self.item_proj = nn.Linear(input_dim, d_model)

        # # Optional: transformer (acts more like attention-based fusion)
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
        #     num_layers=2
        # )

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # Final MLP on fused features
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def project(self, x, layer):
        if x.dim() == 2:
            return layer(x)  # (B, d_model)
        elif x.dim() == 3:
            B, K, D = x.shape
            return layer(x.view(B * K, D)).view(B, K, -1)  # (B, K, d_model)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

    def forward(self, user_emb, item_emb):
        """
        Supports both:
        - Pairwise: user_emb, item_emb of shape (B, D)
        - Listwise: user_emb of shape (B, K, D), item_emb of shape (B, K, D)
        """
        if user_emb.dim() == 2:  # Pairwise: (B, D)
            u = self.user_proj(user_emb).unsqueeze(1)  # (B, 1, D)
            i = self.item_proj(item_emb).unsqueeze(1)  # (B, 1, D)
        elif user_emb.dim() == 3:  # Listwise: (B, K, D)
            B, K, D = user_emb.shape
            u = self.user_proj(user_emb)  # (B, K, D)
            i = self.item_proj(item_emb)  # (B, K, D)
        else:
            raise ValueError("Invalid input dimensions")

        # Form interaction pairs: (B, K, 2, D) → (B*K, 2, D)
        tokens = torch.stack([u, i], dim=2).view(-1, 2, u.shape[-1])  # (B*K, 2, D)

        # MultiheadAttention expects (B, T, D), so split Q, K, V
        query = tokens  # (B*K, 2, D)
        key = tokens
        value = tokens

        attn_out, _ = self.attn(query, key, value)  # (B*K, 2, D)
        u_out, i_out = attn_out[:, 0, :], attn_out[:, 1, :]  # (B*K, D)

        # Fuse via concatenation + interaction features
        fused = torch.cat([
            u_out,
            i_out,
            u_out * i_out,
            torch.abs(u_out - i_out)
        ], dim=-1)  # (B*K, 4D)

        return self.mlp(fused).squeeze(-1)  # (B*K,)

    # def forward(self, user_emb, item_emb):
    #     """
    #     Handles both pairwise and listwise inputs:
    #     - Pairwise: user_emb: (B, D), item_emb: (B, D)
    #     - Listwise: user_emb: (B, K, D), item_emb: (B, K, D)
    #     """

    #     # Project user and item embeddings
    #     user_emb = self.project(user_emb, self.user_proj)
    #     item_emb = self.project(item_emb, self.item_proj)

    #     if user_emb.dim() == 2:
    #         # Pairwise interaction
    #         # Transformer-level interaction
    #         tokens = torch.stack([user_emb, item_emb], dim=1)  # (B, 2, D)
    #         out = self.attn(tokens)      # (B, 2, D)
    #         u_out, i_out = out[:, 0, :], out[:, 1, :] # (B, d_model)

    #         # Fuse with interaction-aware features
    #         features = torch.cat([
    #             u_out,
    #             i_out,
    #             u_out * i_out,
    #             torch.abs(u_out - i_out)
    #         ], dim=-1)  # (B, 4D)

    #         return self.mlp(features).squeeze(-1)

    #     elif user_emb.dim() == 3:
    #         # Listwise mode: assume user_emb is (B, 1, D), item_emb is (B, K, D)
    #         u_expanded = user_emb.expand_as(item_emb)                # (B, K, d_model)
    #         tokens = torch.stack([u_expanded, item_emb], dim=2)  # (B, K, 2, d_model)
    #         B, K, T, D = tokens.shape
    #         tokens = tokens.view(B * K, T, D)           # (B*K, 2, d_model)

    #         out = self.attn(tokens)              # (B*K, 2, d_model)
    #         u_out, i_out = out[:, 0, :], out[:, 1, :]   # (B*K, d_model)

    #         features = torch.cat([u_out, i_out, u_out * i_out, torch.abs(u_out - i_out)], dim=-1)
    #         return self.mlp(features).view(B, K)        # (B, K)

    #     else:
    #         raise ValueError(f"Unexpected input shape: {user_emb.shape}")


