# ranker/data/collate.py

import torch
from torch.nn.utils.rnn import pad_sequence

def listwise_collate(batch):
    """
    Pads item_embs and labels to the max number of candidates in the batch.
    Each batch entry: (user_embs: [K, D], item_embs: [K, D], labels: [K])
    """
    user_embs, item_embs_list, labels_list = zip(*batch)
    max_len = max(x.shape[0] for x in item_embs_list)

    padded_user_embs = []
    padded_item_embs = []
    padded_labels = []

    for u, i, l in zip(user_embs, item_embs_list, labels_list):
        pad_len = max_len - i.shape[0]
        if pad_len > 0:
            pad_tensor_i = torch.zeros((pad_len, i.shape[1]), dtype=i.dtype)
            pad_tensor_u = torch.zeros((pad_len, u.shape[1]), dtype=u.dtype)
            pad_tensor_l = torch.zeros(pad_len, dtype=l.dtype)
            i = torch.cat([i, pad_tensor_i], dim=0)
            u = torch.cat([u, pad_tensor_u], dim=0)
            l = torch.cat([l, pad_tensor_l], dim=0)

        padded_user_embs.append(u)
        padded_item_embs.append(i)
        padded_labels.append(l)

    return (
        torch.stack(padded_user_embs),  # (B, K, D)
        torch.stack(padded_item_embs),  # (B, K, D)
        torch.stack(padded_labels)      # (B, K)
    )


def contextual_collate(batch):
    """
    Collate function for ContextualPairwiseDataset.

    Args:
        batch: list of (history_emb, pos_emb, neg_emb) where
            - history_emb: (H_i, D)
            - pos_emb: (D,)
            - neg_emb: (D,)

    Returns:
        history_padded: (B, H_max, D)
        pos_batch: (B, D)
        neg_batch: (B, D)
        history_lens: list[int]
    """
    history_seqs, pos_list, neg_list = zip(*batch)

    # Ensure all histories are 2D (H, D)
    for i, h in enumerate(history_seqs):
        if h.dim() != 2:
            raise ValueError(f"history_emb at index {i} has invalid shape: {h.shape}")

    history_lens = [h.shape[0] for h in history_seqs]
    max_len = max(history_lens)
    emb_dim = history_seqs[0].shape[1]

    padded_histories = torch.zeros(len(batch), max_len, emb_dim)
    for i, h in enumerate(history_seqs):
        padded_histories[i, :h.shape[0]] = h

    pos_batch = torch.stack(pos_list)  # (B, D)
    neg_batch = torch.stack(neg_list)  # (B, D)

    return padded_histories, pos_batch, neg_batch, history_lens
