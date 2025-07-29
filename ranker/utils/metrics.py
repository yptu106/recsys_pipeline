import torch
import torch.nn.functional as F

def bpr_loss(pos_score, neg_score):
    return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

def listwise_loss(pred_scores, labels):
    """
    pred_scores: (B*K,) flattened predicted scores
    labels: (B*K,) flattened labels
    Assumes scores and labels are grouped per user (e.g. batch size B with K candidates per user)
    """
    B = labels.shape[0]
    # Infer K from user batch (assumes uniform K)
    assert pred_scores.dim() == 1, "Expected flat (B*K,) prediction tensor"
    K = (labels != -100).sum() // B  # or pass K explicitly if known

    pred_scores = pred_scores.view(B, K)
    labels = labels.view(B, K)

    pred_prob = F.softmax(pred_scores, dim=1)
    label_prob = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)

    return F.kl_div(pred_prob.log(), label_prob, reduction='batchmean')