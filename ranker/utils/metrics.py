import torch

def bpr_loss(pos_score, neg_score):
    return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))
