# Project
A multi-stage recommender system pipeline (retrieval -> ranking -> re-ranking) for livestreaming interactions. 

# Directory Layout

```
├── data
│   ├── atomic          # atomic files for RecBole
│   ├── processed       # processed interactions data
│   ├── raw             # raw dataset
│   └── splits          # train, validation, testing
├── embeddings          # generated embeddings
├── env             
├── features            # item_sentence, aggregated user/item features
├── index               # FAISS indices
├── Makefile
├── README.md
├── results             # retrieval, ranking, reranking outputs
└── src                 # source code and checkpoints
    ├── config.py     
    ├── encoder         # retrieval encoder
    ├── eval            # evaluation scripts
    ├── indexing        # build FAISS indices  
    ├── preprocessing   # scripts for preprocessing dataset
    ├── ranker          # torch & RecBole ranker
    ├── representations # wrapper to map id into its corresponding embedding
    ├── reranker        # re-ranker object
    └── retrieval       # wrapper to run two-tower retrieval
```

# Usage
## Run the full pipeline
```bash
make all
```
This will run through preprocssing -> retrieval -> ranking -> reranking. 

Note that you may need to change environment from `ml_env` to `recbole` to run `BPR` and `SASRec` ranker. 

## Run individual stages
```bash
make build_interactions
make build_item_sentence
make build_streamer_emb
make build_index
make split
make retrieve
make rank
make rerank
```

## Evaluation
Evaluate different stages:
```bash
make eval_retrieval
make eval_ranking
make eval_reranking
```
Outputs metrics on:
* Full dataset
* Repeat interactions
* Novel interactions

## Configurable Parameters (Makefile knobs)
You can override defaults via CLI, e.g., `make rank RANKER_MODEL=SASRec`.

| Variable            | Options                            | Default                           | Description                 |
| ------------------- | ---------------------------------- | --------------------------------- | --------------------------- |
| `DATASET`           | livestream                         | livestream                        | Dataset name                |
| `DATA_VERSION`      | YYYY-MM-DD                         | 2025-06-30                        | Data version                |
| `INTERACTION_TYPE`  | donate \| enter                    | donate                            | Type of user interaction    |
| `EMB_COL`           | item\_sentence \| format\_sentence | item\_sentence                    | Feature column for encoding |
| `RETRIEVAL_ENCODER` | MiniLM \| bge \| others            | MiniLM                            | Encoder backbone            |
| `INDEX_TYPE`        | flat \| hnsw \| ivf                | flat                              | FAISS index type            |
| `RANKER_MODEL`      | BPR \| SASRec                      | BPR                               | Ranking model               |
| `RERANK_STRATEGY`   | mmr \| others                      | mmr                               | Re-ranking strategy         |
| `SPLIT_TYPE`        | time \| random                     | time                              | Train/test split strategy   |
| `FILTER`            | heavy\_filter\_missing\_streamers  | heavy\_filter\_missing\_streamers | Filtering strategy          |

### Directory Naming
Currently, we use Split ID and Retrieval Profile to manage the artifacts of each stage. 
* Split ID (`SPLIT_ID`) = `INTERACTION_TYPE`_`SPLIT_TYPE`_`FILTER`
    * Example: `donate_time_heavy_filter_missing_streamers`
* Retrieval Profile (`RETR_PROFILE`) = `EMB_COL`_`RETRIEVAL_ENCODER`_`INDEX_TYPE`
    * Example: `item_sentence_MiniLM_flat`

### Notes on `FILTER`
* The `FILTER` variable only affects the directory ID (for naming consistency).
* Filtering logic (e.g., `--filter-missing-streamers`, `--filter-too-few-streamers`) is currently hardcoded in Makefile. 
* Changing `FILTER` alone will not alter preprocessing behavior. Check of Makefile for more details. 

## Notes on RecBole Ranker
* The RecBole package requires specific versions of PyTorch and NumPy. You may need to activate the `recbole` environment to run the ranker.
* Since we hardcoded the dataset directory for training RecBole models in the config files, the atomic files required by the model do not follow the directory naming convention based on Split ID and Retrieval Profile. Future work may need to refactor this part.
* If you'd like to run `SASRecF` to incorporate item embeddings, you'll need to apply our local bug fix to RecBole, follow these steps:

```bash
# Clone the official RecBole repo
git clone https://github.com/RUCAIBox/RecBole.git
cd RecBole

# Apply the diff stored in this repository
git apply ../src/ranker/recbole/recbole_fix.diff

# Install the patched version locally
pip install -e .

```
After installation, all ranker scripts will use the patched RecBole instead of the remote PyPI package. 

Please refer to the following link for the details:
https://github.com/RUCAIBox/RecBole/issues/2104

## Checkpoints
* Checkpoints for two-tower retrieval encoder are stored under `/nas02/home/kevin/recsys_pipeline/src/encoder/checkpoints/epoch40`. 
* Checkpoints for RecBole ranker are stored under `/nas02/home/kevin/recsys_pipeline/src/ranker/recbole/checkpoints`. 
* Checkpoints for torch ranker are stored under `/nas02/home/kevin/recsys_pipeline/src/ranker/torch/checkpoints`. 


## Raw Dataset
Please refer to `/nas02/home/kevin/recsys_pipeline/data/raw/livestream/2025-06-30` for the raw dataset used in this experiement. 