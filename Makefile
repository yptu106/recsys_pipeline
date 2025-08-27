# Default goal (what runs if you just type "make")
SHELL := /bin/bash
.DEFAULT_GOAL := all
.SHELLFLAGS := -eu -o pipefail -c

# ---- knobs (override on CLI) ----
DATASET ?= livestream
DATA_VERSION ?= 2025-06-30
# donate | enter
INTERACTION_TYPE ?= donate
# item_sentence | format_sentence
EMB_COL ?= item_sentence
# MiniLM | bge | others
RETRIEVAL_ENCODER ?= MiniLM
# flat | hnsw | ivf
INDEX_TYPE ?= flat
# BPR | SASRec | others
RANKER_MODEL ?= BPR
# mmr | others
RERANK_STRATEGY ?= mmr

SPLIT_TYPE ?= time
FILTER ?= heavy_filter_missing_streamers

# map retrieval model to its full name and checkpoint
ifeq ($(RETRIEVAL_ENCODER),MiniLM)
	ENCODER := sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
	ENCODER_CKPT := src/encoder/checkpoints/epoch40/best_model
else ifeq ($(RETRIEVAL_ENCODER),bge)
	ENCODER := bge-m3
# 	ENCODER_CKPT := NONE
else
	$(error "Unknown retrieval encoder: $(RETRIEVAL_ENCODER)")
endif

# map split type to its strategy name
ifeq ($(SPLIT_TYPE),time)
  SPLIT_STRATEGY := time_based
else ifeq ($(SPLIT_TYPE),random)
  SPLIT_STRATEGY := uniform_random_lko
else
  $(error Unknown split type: $(SPLIT_TYPE))
endif

# map ranker model to its checkpoint, config
ifeq ($(RANKER_MODEL),BPR)
	RANKER := BPR
	RECBOLE_CFG := src/ranker/recbole/configs/BPR/ts_heavy.yaml
	RECBOLE_CKPT := src/ranker/recbole/checkpoints/BPR/ts_heavy/latest.pth
else ifeq ($(RANKER_MODEL),SASRec)
	RANKER := SASRec
	RECBOLE_CFG := src/ranker/recbole/configs/SASRec/ts_heavy.yaml
	RECBOLE_CKPT := src/ranker/recbole/checkpoints/SASRec/ts_heavy/latest.pth
else
	$(error "Unknown ranker model: $(RANKER_MODEL)")
endif



# ---- construct profile ids ----
SPLIT_ID := $(INTERACTION_TYPE)_$(SPLIT_TYPE)_$(FILTER)
RETR_PROFILE := ${EMB_COL}_$(RETRIEVAL_ENCODER)_$(INDEX_TYPE)

# ---- paths ----
RAW_DIR        := data/raw/$(DATASET)/$(DATA_VERSION)
PROC_DIR       := data/processed/$(DATASET)/$(DATA_VERSION)/$(INTERACTION_TYPE)
SPLIT_DIR      := data/splits/$(DATASET)/$(DATA_VERSION)/$(INTERACTION_TYPE)/$(SPLIT_ID)
FEAT_DIR       := features/$(DATASET)/$(DATA_VERSION)/$(SPLIT_ID)
EMB_DIR        := embeddings/$(DATASET)/$(DATA_VERSION)/$(SPLIT_ID)/$(RETR_PROFILE)
IDX_DIR        := index/faiss/$(DATASET)/$(DATA_VERSION)/$(SPLIT_ID)/$(RETR_PROFILE)
RES_BASE       := results/$(DATASET)/$(DATA_VERSION)/$(SPLIT_ID)/$(RETR_PROFILE)
RET_DIR        := $(RES_BASE)/retrieved
RANK_DIR       := $(RES_BASE)/ranked/$(RANKER)
RERANK_DIR     := $(RES_BASE)/reranked/$(RANKER)/$(RERANK)

# -------- artifacts --------
PROC_OUT       := $(PROC_DIR)/latest.parquet

ITEM_SENT_OUT  := $(FEAT_DIR)/item_sentence/latest.parquet
STREAMER_FEAT  := $(FEAT_DIR)/streamer/latest.parquet
USER_FEAT_OUT  := $(FEAT_DIR)/user/latest.parquet

EMB_STREAMER   := $(EMB_DIR)/streamer/embeddings.npy
IDX_STAMP      := $(IDX_DIR)/index.idx
SPLIT_STAMP    := $(SPLIT_DIR)/.split_done

RETRIEVE_STAMP := $(RET_DIR)/.retrieve_done
RANK_STAMP     := $(RANK_DIR)/.rank_done
RERANK_STAMP   := $(RERANK_DIR)/.rerank_done

# ---- tools ----
PY := python
BUILD_INTERACTIONS := -m src.preprocessing.build_interactions
BUILD_ITEM_SENTENCE := -m src.preprocessing.build_item_sentence
BUILD_AGG_FEATURES := -m src.preprocessing.build_aggregate_features
SPLIT := -m src.preprocessing.split_dataset
BUILD_EMB_STREAMER := -m src.encoder.build_streamer_emb

# ---- diectories to ensure exist ----
NEEDED_DIRS := \
	$(RAW_DIR) \
	$(PROC_DIR) \
	$(FEAT_DIR)/item_sentence $(FEAT_DIR)/streamer $(FEAT_DIR)/user \
	$(EMB_DIR)/streamer $(EMB_DIR)/user \
	$(IDX_DIR) $(SPLIT_DIR) $(RET_DIR) $(RANK_DIR) $(RERANK_DIR)

# ---- toy targets that only print paths ----
.PHONY: all build_item_sentence streamer_emb index preprocess split agg retrieve rank rerank echo-vars

# Run the whole pipeline
all: rerank

%/:
	@mkdir -p $@

dirs:
	@mkdir -p $(NEEDED_DIRS)

# 1) preprocess interactions (filter by interaction type) 
$(PROC_OUT): | dirs
	@echo "› build_interactions -> $(PROC_OUT)"
	$(PY) $(BUILD_INTERACTIONS) \
	  --csv $(RAW_DIR)/interactions.csv \
	  --filter-conditions $(INTERACTION_TYPE) \
	  --out-dir $(PROC_DIR)

build_interactions: $(PROC_OUT)

# 2) build item sentences
$(ITEM_SENT_OUT): $(PROC_OUT)
	@echo "› build_item_sentence -> $(ITEM_SENT_OUT)"
	$(PY) $(BUILD_ITEM_SENTENCE) \
	  --streamers-csv $(RAW_DIR)/streamers.csv \
	  --out-dir $(FEAT_DIR)/item_sentence

build_item_sentence: $(ITEM_SENT_OUT)

# 3) streamer embeddings
$(EMB_STREAMER): $(ITEM_SENT_OUT)
	@echo "› streamer_embeddings -> $(EMB_STREAMER)"
	$(PY) $(BUILD_EMB_STREAMER) \
	  --features $(ITEM_SENT_OUT) \
	  --encode-col $(EMB_COL) \
	  --model $(ENCODER) \
	  --model-path $(ENCODER_CKPT) \
	  --out-dir $(EMB_DIR)/streamer \
	  --normalize

build_streamer_emb: $(EMB_STREAMER)

# 4) build index
$(IDX_STAMP): $(EMB_STREAMER)
	@echo "› build_index -> $(IDX_STAMP)"
	$(PY) -m src.indexing.build_faiss \
	  --embeddings $(EMB_STREAMER) \
	  --index-type $(INDEX_TYPE) \
	  --out $(IDX_STAMP) \

build_index: $(IDX_STAMP)

# 5) split dataset
$(SPLIT_STAMP): $(PROC_OUT)
	@echo "› split -> $(SPLIT_DIR)"

	$(PY) -m src.preprocessing.split_dataset \
	  --interactions $(PROC_OUT) \
	  --filter-missing-streamers \
	  --streamer-lookup $(EMB_DIR)/streamer/lookup.parquet \
	  --filter-too-few-streamers \
	  --split-repeat-novel \
	  --strategy $(SPLIT_STRATEGY) \
	  --out-dir $(SPLIT_DIR) \
	  --neg_per_pos 5 \
	  --val_k 1 \
	  --test_k 1
	@touch $(SPLIT_STAMP)

split: $(SPLIT_STAMP)

# 6) run retrieval
.PHONY: retrieve

retrieve: $(SPLIT_STAMP) $(IDX_STAMP) $(EMB_STREAMER) | $(RET_DIR)/
	@echo "› retrieve -> $(RET_DIR)"
	$(PY) -m src.retrieval.run_retriever \
	  --test-path $(SPLIT_DIR)/test.parquet \
	  --emb-dir $(EMB_DIR)/streamer \
	  --index $(IDX_STAMP) \
	  --user-log $(SPLIT_DIR)/interactions_train.parquet \
	  --out-dir $(RET_DIR) \
	  --k 500
	@touch $(RETRIEVE_STAMP)

# 7) run ranking
$(RANK_STAMP): $(RETRIEVE_STAMP) | $(RANK_DIR)/
	@echo "› rank -> $(RANK_DIR)"
	$(PY) -m src.ranker.recbole.run_ranker \
	  --test-path $(SPLIT_DIR)/test.parquet \
	  --retrieval-dir $(RET_DIR) \
	  --out-dir $(RANK_DIR) \
	  --model $(RANKER) \
	  --recbole-config-path $(RECBOLE_CFG) \
	  --checkpoint-path $(RECBOLE_CKPT) \
	  --topk 50

	@touch $(RANK_STAMP)

.PHONY: rank
rank: $(RANK_STAMP)

# 8) run reranking
$(RERANK_STAMP): $(RANK_STAMP) | $(RERANK_DIR)/
	@echo "› rerank -> $(RERANK_DIR)"
	$(PY) -m src.reranker.run_reranker \
	  --test-path $(SPLIT_DIR)/test.parquet \
	  --strategy $(RERANK_STRATEGY) \
	  --lambda_ 0.3 \
	  --embedding-dir $(EMB_DIR)/streamer \
	  --ranked-dir $(RANK_DIR) \
	  --topk 100 \
	  --out-dir $(RERANK_DIR)

	@touch $(RERANK_STAMP)

.PHONY: rerank
rerank: $(RERANK_STAMP)


# evaluate retrieval reuslts
.PHONY: eval_retrieval
eval_retrieval: $(RETRIEVE_STAMP)
	@echo "› eval_retrieval -> $(RET_DIR)"
	@echo "› full dataset"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/test.parquet \
	  --dir $(RET_DIR) \
	  --ks 10 20 50 100 500
	
	@echo "› repeat set"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/repeat_novel/repeat.parquet \
	  --dir $(RET_DIR) \
	  --ks 10 20 50 100 500

	@echo "› novel set"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/repeat_novel/novel.parquet \
	  --dir $(RET_DIR) \
	  --ks 10 20 50 100 500


.PHONY: eval_ranking
eval_ranking: $(RANK_STAMP)
	@echo "› eval_ranking -> $(RANK_DIR)"
	@echo "› full dataset"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/test.parquet \
	  --dir $(RANK_DIR) \
	  --ks 10 20 50 100
	
	@echo "› repeat set"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/repeat_novel/repeat.parquet \
	  --dir $(RANK_DIR) \
	  --ks 10 20 50 100

	@echo "› novel set"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/repeat_novel/novel.parquet \
	  --dir $(RANK_DIR) \
	  --ks 10 20 50 100

.PHONY: eval_reranking
eval_reranking: $(RERANK_STAMP)
	@echo "› eval_reranking -> $(RERANK_DIR)"
	@echo "› full dataset"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/test.parquet \
	  --dir $(RERANK_DIR) \
	  --ks 10 20 50 100
	
	@echo "› repeat set"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/repeat_novel/repeat.parquet \
	  --dir $(RERANK_DIR) \
	  --ks 10 20 50 100

	@echo "› novel set"
	$(PY) -m src.eval.evaluate \
	  --test-path $(SPLIT_DIR)/repeat_novel/novel.parquet \
	  --dir $(RERANK_DIR) \
	  --ks 10 20 50 100