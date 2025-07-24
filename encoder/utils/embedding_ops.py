import torch
from collections import defaultdict

def build_user_embedding_cache(encoder, user_to_items, item_id_to_text, batch_size=256):
    """Builds a cache of user embeddings by averaging item embeddings."""
    all_texts = []
    all_user_ids = []

    for user_id, item_ids in user_to_items.items():
        for item_id in item_ids:
            if item_id in item_id_to_text:
                all_texts.append(item_id_to_text[item_id]) # item text
                all_user_ids.append(user_id)               # user_id owning the item text

    print(f"› Encoding {len(all_texts)} item texts for {len(user_to_items)} users...")
    all_embeddings = encoder.encode(all_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)

    emb_sum = defaultdict(lambda: torch.zeros_like(all_embeddings[0]))
    emb_count = defaultdict(int)

    for uid, emb in zip(all_user_ids, all_embeddings):
        emb_sum[uid] += emb
        emb_count[uid] += 1

    user_embeddings = {uid: emb_sum[uid] / emb_count[uid] for uid in emb_sum}
    return user_embeddings

def encode_texts(model, texts, device):
    tokens = model.tokenize(texts)  # {'input_ids': ..., 'attention_mask': ...}
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.set_grad_enabled(True):
        outputs = model(tokens)
    return outputs['sentence_embedding']  # [batch_size, hidden_dim]