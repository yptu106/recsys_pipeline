import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualUserEncoder(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.item_proj = nn.Linear(input_dim, proj_dim)
        self.history_proj = nn.Linear(input_dim, proj_dim)

    def forward(self, item_emb, user_history_emb):
        """
        Args:
            item_emb: (B, D) — the candidate item embedding
            user_history_emb: (B, H, D) — sequence of past interacted item embeddings
                - B: batch size
                - H: length of user history (number of past items)
                - D: embedding dimension
        
        Returns:
            u_context: (B, D_proj) — contextual user embedding
        """
        q = self.item_proj(item_emb).unsqueeze(1)     # (B, 1, D_proj)
        k = self.history_proj(user_history_emb)        # (B, H, D_proj)

        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)  # (B, 1, H)
        u_context = torch.bmm(attn_weights, k)  # (B, 1, D_proj)
        return u_context.squeeze(1)  # (B, D_proj)

class ContextualRanker(nn.Module):
    def __init__(self, input_dim, proj_dim=256):
        super().__init__()
        self.contextual_encoder = ContextualUserEncoder(input_dim=input_dim, proj_dim=proj_dim)
        self.item_proj = nn.Linear(input_dim, proj_dim)

        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, user_history_emb, item_emb):
        """
        user_history_emb: (B, N, D)
        item_emb: (B, D)
        """
        # print("[contextual ranker] user_history_emb:", user_history_emb.shape)
        user_context = self.contextual_encoder(item_emb, user_history_emb)  # (B, D)
        # print("[contextual ranker] user_context:", user_context.shape)

        # print("[contextual ranker] item_emb (before proj):", item_emb.shape)
        item_proj = self.item_proj(item_emb)  # (B, D)
        # print("[contextual ranker] item_emb (after proj):", item_proj.shape)

        fused = torch.cat([
            user_context,
            item_proj,
            user_context * item_proj,
            torch.abs(user_context - item_proj)
        ], dim=-1)  # (B, 4D)

        return self.mlp(fused).squeeze(-1)  # (B,)
