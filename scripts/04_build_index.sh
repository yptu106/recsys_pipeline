
#!/bin/bash
set -e

emb_type="$1" # item_sentence or format_sentence or item_sentence_num or format_sentence_num
model_name="$2" # MiniLM or bge
index_type="$3" # flat or hnsw


python -m src.vector_store.build_faiss \
    --embeddings embeddings/$model_name/$emb_type/streamer_embeddings.npy \
    --out index/faiss/$model_name/$emb_type/index_${index_type}.idx \
    --index-type $index_type