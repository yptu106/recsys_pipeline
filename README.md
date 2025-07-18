
# Workflow
All scripts are expected to be run from the project root directory, unless otherwise specified. 

## 1. Preprocessing
```
# Generate filtered interactions
./scripts/01_build_interactions.sh data/raw/interactions.csv donate

# Build streamer-level features
./scripts/02_build_streamer_features.sh data/raw/streamers.csv donate

```

## 2. Embedding + Indexing
```
# Generate streamer sentence embeddings
./scripts/03_build_streamer_embeddings.sh features/streamer/latest.parquet item_sentence true MiniLM

# Build FAISS index for retrieval
./scripts/04_build_index.sh item_sentence_num MiniLM flat

```

## 3. Data Split
```
# Create train/val/test sets
./scripts/05_split_dataset.sh donate embeddings/streamer/MiniLM/item_sentence_num/lookup.parquet

```

## 4. Retrieval & Ranking
To retrieve and rank streamers for a single user_id, use the scripts under src/services/.

```
python -m src.services.retrieval \
    --user-id 1000015 \
    --emb-dir embeddings/streamer/MiniLM/item_sentence_num \
    --index index/faiss/MiniLM/item_sentence_num/index_flat.idx \
    --user-log data/splits/donate/interactions_train.parquet \
    --out-dir results/retrieval/MiniLM/item_sentence_num

python -m src.services.rank \
    --user-id 1000015 \
    --retrieval-path results/retrieval/MiniLM/item_sentence_num/user_1000015.json \
    --feature-dir features/ranker/lightgbm \
    --model-dir ranker/models \
    --topk 100 \
    --out-dir results/ranked/lightgbm
```

## 5. Evaluation
To evaluate the performance (e.g., Recall@K, MRR, nDCG) over all sampled users in the test split, use the scripts in scripts/eval/.

```
# Run retrieval for all users
./scripts/eval/01_run_retrieval.sh donate item_sentence_num MiniLM 500

# Evaluate retrieval result
./scripts/eval/02_eval_retrieval.sh donate item_sentence_num MiniLM

# Run ranking for all users
./scripts/eval/03_run_rank.sh donate item_sentence_num MiniLM 100

# Evaluate ranking result
./scripts/eval/04_eval_rank.sh donate lightgbm
```


# Directory Structure
```
├── data/              # Raw, processed, and split datasets
├── embeddings/        # Streamer/user embeddings (e.g., MiniLM)
├── env/               # Conda environment configuration
├── features/          # Streamer features (e.g., watch_ts, flattened sentence)
├── notebooks/         # EDA and preprocessing notebooks
├── ranker/            # LightGBM and MLP rankers
├── scripts/           # Shell scripts for each stage of the pipeline
├── src/               # Core Python modules (retrieval, ranking, eval, etc.)
```

## Preprocessing
Converts raw interaction and streamer metadata CSVs into filtered Parquet files. Also generates derived statistics and streamer description fields to be used in downstream embedding and ranking modules.

Location: `src/preprocessing/`

| Script                  | Purpose                                                                                   |
|-------------------------|-------------------------------------------------------------------------------------------|
| `build_interactions.py` | Loads and filters raw user-streamer interaction data; outputs filtered interactions in Parquet format |
| `build_features.py`     | Generates streamer-level features (e.g., total watch time, follower count, gift metrics) and streamer description based on meta data|
| `split_dataset.py`      | Splits interaction data into train/val/test sets based leave-one-out random sampling strategy                      |
| `preprocess_top100.ipynb` | Jupyter notebook for manually reviewing or preprocessing the top 100 streamers list     |

## Embeddings
Generates sentence-based streamer embeddings using transformer models (e.g., MiniLM), and constructs FAISS indexes for retrieval. 

| Script                  | Purpose                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `build_streamer_emb.py` | Loads streamer metadata and generates sentence embeddings using a transformer model              |
| `train_encoder.py`      | *Work in Progress* Trains a transformer-based encoder (e.g., using sentence-transformers) on streaming metadata     |

## Ranker
Contains the source code and training scripts for second-stage ranking. This module is used after retrieval to rerank the top-N candidates.

Location: `ranker/`

| Path / Script                | Purpose                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| `lightgbm/train_lightgbm.py` | Trains a LightGBM ranker using feature-engineered user–streamer pairs                   |
| `lightgbm/build_features.py` | Builds ranking features (e.g., watch counts, follow/gift history) for LightGBM training |
| `mlp/train_mlp.py`           | Trains a neural network ranker (MLP) on user-positive-negative triplets        |
| `mlp/mlp_ranker.py`          | Defines the architecture and forward pass for the MLP ranker                            |
| `mlp/pairwise_dataset.py`    | Constructs PyTorch `Dataset` objects for triplet-based training of the MLP              |


## Services
Implements model inference services for retrieval and ranking. These scripts are designed to be used during evaluation or deployment stages and serve as the entry points for pipeline components.

Location: `src/services/`

| Script              | Purpose                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| `retrieval.py`      | Loads FAISS index and performs nearest neighbor search to retrieve top-k candidate streamers per user    |
| `build_user_emb.py` | Precompute user embeddings by averaging the embeddings of previously watched streamers         |
| `rank.py`           | LightGBM model to score and rank retrieved items |
| `rank_pop.py`       | Simple popularity-based ranking logic for use as a baseline or fallback method                         |
| `rank_mlp.py`       | Loads and runs MLP model to score and rank retrieved items                                             |
