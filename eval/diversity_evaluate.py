import argparse
import json
import pathlib
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import Counter
from math import log
from typing import List, Dict, Tuple, Union

seed = 42
random.seed(seed)
np.random.seed(seed)

def load_lists(dir_path: Union[str, pathlib.Path], K: int) -> Dict[int, List[int]]:
    dir_path = pathlib.Path(dir_path)
    out = {}
    for p in dir_path.glob("user_*.json"):
        uid = int(p.stem.split("_")[1])
        items = [int(r["streamer_id"]) for r in json.loads(p.read_text())][:K]
        out[uid] = items
    return out

def sample_pairs(uids: List[int], m: int) -> List[Tuple[int, int]]:
    """
    Sample unique user pairs from a list of user IDs.

    Args:
        uids (List[int]): List of user IDs.
        m (int): Number of pairs to sample.

    Returns:
        List[Tuple[int, int]]: List of sampled user pairs.
    """
    if len(uids) < 2: 
        return []
    seen, out = set(), []
    max_pairs = (len(uids) * (len(uids) - 1)) // 2
    target = min(m, max_pairs)
    while len(out) < target:
        i, j = random.sample(uids, 2)
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in seen:
            seen.add((a, b)); out.append((a, b))
    return out

# ----------------- pairwise similarity -----------------

def jaccard(a: List[int], b: List[int]) -> float:
    """
    Calculate Jaccard similarity between two lists.
    Jaccard similarity is defined as the size of the intersection divided by the size of the union.

    Args:
        a (List[int]): First list of items.
        b (List[int]): Second list of items.

    Returns:
        float: Jaccard similarity score.
    """
    A, B = set(a), set(b)
    u = len(A | B)
    return 0.0 if u == 0 else len(A & B) / u

def rbo(user_list_a: List[int], user_list_b: List[int], p: float = 0.9) -> float:
    """
    Calculate Rank-Biased Overlap (RBO) between two ranked lists.

    Args:
        user_list_a (List[int]): First ranked list.
        user_list_b (List[int]): Second ranked list.
        p (float): Persistence parameter (0.8–0.98 typical).

    Returns:
        float: RBO score.
    """
    # Simple equal-length RBO (top-heavy)
    list_a = user_list_a[:]
    list_b = user_list_b[:]

    seen_a, seen_b = set(), set()
    running_sum = 0.0
    for depth, (item_a, item_b) in enumerate(zip(list_a, list_b), start=1):
        seen_a.add(item_a)
        seen_b.add(item_b)
        overlap_prefix = len(seen_a & seen_b) / depth
        running_sum += overlap_prefix * (p ** (depth - 1))
    return (1 - p) * running_sum

# ----------------- exposure / concentration -----------------

def exposure_stats(lists: Dict[int, List[int]]):
    """
    Calculate exposure statistics.

    Args:
        lists (Dict[int, List[int]]): Dictionary mapping user IDs to lists of items.

    Returns:
        Dict[str, float]: Dictionary containing exposure statistics:
            - coverage_items: Number of unique items shown.
            - gini: Gini coefficient for item exposure.
            - hhi: Herfindahl-Hirschman Index for item exposure.
            - entropy: Normalized entropy of item exposure.
    """
    expo = Counter(i for L in lists.values() for i in L)
    n = sum(expo.values())
    if n == 0:
        return {"coverage_items": 0, "gini": 0.0, "hhi": 0.0, "entropy": 0.0}
    xs = np.array(list(expo.values()), dtype=np.float64)
    xs.sort()
    i = np.arange(1, len(xs) + 1)
    gini = (np.sum((2*i - len(xs) - 1) * xs) / (len(xs) * np.sum(xs))) if len(xs) else 0.0
    p = xs / n
    hhi = float(np.sum(p**2))
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    ent_norm = ent / log(len(xs)) if len(xs) > 1 else 0.0
    cov = len(expo)  # unique items shown; divide by catalog size upstream if you have it
    return {"coverage_items": int(cov), "gini": float(gini), "hhi": hhi, "entropy": ent_norm}

# ----------------- summarization & reporting -----------------

def summarize(values: List[float]) -> Dict[str, float]:
    """
    Summarize a list of float values.
    """
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p25": np.nan, "p75": np.nan, "p90": np.nan}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }

def list_duplication_rate(lists: Dict[int, List[int]]) -> float:
    """
    Calculate the duplication rate of lists.
    It measures the proportion of users with any duplicate lists of items.

    Args:
        lists (Dict[int, List[int]]): Dictionary mapping user IDs to lists of items.

    Returns:
        float: Duplication rate, defined as the proportion of users with duplicate lists.
    """
    # Convert lists to a string representation for counting duplicates
    # e.g., {"user1": [1, 2, 3], "user2": [1, 2, 3]} -> ["1|2|3", "1|2|3"]
    seqs = ["|".join(map(str, v)) for v in lists.values()]
    if not seqs: 
        return 0.0

    counts = Counter(seqs)
    dup_users = sum(c for c in counts.values() if c > 1)

    return dup_users / len(seqs)

def mode_list_share(lists: Dict[int, List[int]]) -> float:
    """
    Calculate the mode list share, defined as the proportion of users with the most common list.
    A mode list is the most frequently occurring list of items across all users.
    It measures the proportion of users with the most common list of items.

    Args:
        lists (Dict[int, List[int]]): Dictionary mapping user IDs to lists of items.
        
    Returns:
        float: Proportion of users with the most common list.
    """
    # Convert lists to a string representation for counting duplicates
    # e.g., {"user1": [1, 2, 3], "user2": [1, 2, 3]} -> ["1|2|3", "1|2|3"]
    seqs = ["|".join(map(str, v)) for v in lists.values()]
    if not seqs: 
        return 0.0

    counts = Counter(seqs)

    return max(counts.values()) / len(seqs)

def personalization_overview(lists: Dict[int, List[int]], K: int, pair_samples: int, rbo_p: float):
    """
    Generate a personalization overview report for a set of user lists.
    """
    users = list(lists.keys())

    # randomly sample user pairs for pairwise similarity
    pairs = sample_pairs(users, pair_samples)
    if not pairs:
        print("No unique user pairs found. Ensure there are enough users in the candidate directory.")
        return
    
    jaccard_scores = []
    rbo_scores = []
    for u, v in tqdm(pairs, desc="Calculating Jaccard and RBO scores"):
        u_items = lists[u]
        v_items = lists[v]
        jaccard_scores.append(jaccard(u_items, v_items))
        rbo_scores.append(rbo(u_items, v_items, p=rbo_p))

    exposure = exposure_stats(lists)

    report = {
        "users": len(users),
        "pair_samples": len(pairs),
        "K": K,
        # Pairwise similarity summaries
        "jaccard": summarize(jaccard_scores),
        "rbo": summarize(rbo_scores),
        # “Personalization” = 1 − similarity (higher is better personalization)
        "personalization_jaccard_mean": (1 - float(np.mean(jaccard_scores))) if jaccard_scores else np.nan,
        "personalization_rbo_mean": (1 - float(np.mean(rbo_scores))) if rbo_scores else np.nan,
        # Duplication diagnostics
        "mode_list_share": mode_list_share(lists),
        "list_duplication_rate": list_duplication_rate(lists),
        # Exposure concentration
        "exposure": exposure,
    }

    return report


def pretty_print_report(r: Dict):
    print(f"# Users: {r['users']}")
    print(f"# Pair samples: {r['pair_samples']}  (K={r['K']})")
    print("\nPairwise Jaccard:")
    for k, v in r["jaccard"].items():
        print(f"  {k:>7}: {v:.4f}")
    print("Pairwise RBO:")
    for k, v in r["rbo"].items():
        print(f"  {k:>7}: {v:.4f}")
    print(f"\nPersonalization (1 - similarity):")
    print(f"  mean (Jaccard): {r['personalization_jaccard_mean']:.4f}")
    print(f"  mean (RBO)    : {r['personalization_rbo_mean']:.4f}")
    print("\nDuplication:")
    print(f"  mode_list_share     : {r['mode_list_share']:.4f}")
    print(f"  list_duplication_rate: {r['list_duplication_rate']:.4f}")
    print("\nExposure (across all users' top-K):")
    exp = r["exposure"]
    print(f"  coverage_items: {exp['coverage_items']}")
    print(f"  gini          : {exp['gini']:.4f}")
    print(f"  hhi           : {exp['hhi']:.4f}")
    print(f"  entropy       : {exp['entropy']:.4f}")

# ----------------- global bias (optional) -----------------

def load_global_list(path: Union[str, pathlib.Path], max_n: int = 100, item_col: str = "streamer_id") -> List[int]:
    """
    Load a global list of items from a JSON file, limiting to top-K items.
    This global list can be used to compute the fraction of each user's top-K that are in the global list.

    Args:
        path (Union[str, pathlib.Path]): Path to the JSON file.
        K (int): Number of top items to return.
        item_col (str): Column name for item IDs in the JSON data.

    Returns:
        List[int]: List of top-K item IDs.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Global list file not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found in the global list.")
    
    # Get top-K items based on frequency
    items = df[item_col].to_list()

    return items[:max_n] if len(items) > max_n else items

def dcg_at_k(relevance_labels: np.ndarray, top_k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.
    `relevance_labels` should be a 1D array where each entry is a graded or binary relevance.
    """
    relevance_labels = relevance_labels[:top_k]
    if relevance_labels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevance_labels.size + 2))
    return float((relevance_labels * discounts).sum())


def global_ndcg_at_k(user_ranked_items: List[int], global_popular_item_set: set[int], top_k: int) -> float:
    """
    Rank-aware overlap with the global popularity set.
    Relevance is 1 if the item is in the global set, else 0.
    """
    relevance_labels = np.array(
        [1.0 if item_id in global_popular_item_set else 0.0 for item_id in user_ranked_items[:top_k]],
        dtype=np.float32,
    )
    dcg = dcg_at_k(relevance_labels, top_k)
    ideal_relevance = np.ones((min(top_k, len(global_popular_item_set)),), dtype=np.float32)
    ideal_dcg = dcg_at_k(ideal_relevance, min(top_k, len(global_popular_item_set)))
    return 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg


def global_bias_overview(user_top_k_lists: Dict[int, List[int]], global_ordered_items: List[int], top_k: int, rbo_p: float):
    users = list(user_top_k_lists.keys())

    global_ordered_items = global_ordered_items[:top_k]
    global_item_set = set(global_ordered_items)

    overlap_fractions, global_ndcgs, rbo_values = [], [], []
    position_hits = np.zeros(top_k, dtype=np.int64)
    total_global_exposures = 0

    exact_sequence_matches = 0
    exact_set_matches = 0
    global_sequence_tuple = tuple(global_ordered_items)
    global_set_top_k = set(global_ordered_items[:top_k])

    for user_id in tqdm(users, desc="Calculating global bias metrics"):
        ranked_items = user_top_k_lists[user_id][:top_k]

        # fraction of the user's top-K that are in the global list
        overlap_fractions.append(len(set(ranked_items) & global_item_set) / float(len(ranked_items) or 1))

        # nDCG based on global list
        global_ndcgs.append(global_ndcg_at_k(ranked_items, global_item_set, top_k))

        # order-aware similarity to the global order
        rbo_values.append(rbo(ranked_items, global_ordered_items, p=rbo_p))

        # position-wise presence of global items
        for position, item_id in enumerate(ranked_items):
            if item_id in global_item_set:
                position_hits[position] += 1
                total_global_exposures += 1

        # exact matches
        if tuple(ranked_items) == global_sequence_tuple:
            exact_sequence_matches += 1
        if set(ranked_items) == global_set_top_k:
            exact_set_matches += 1
    
    num_users = len(users)
    total_impressions = num_users * top_k
    position_global_share = (position_hits / max(1, num_users)).astype(np.float64)
    exposure_share_global = total_global_exposures / float(total_impressions or 1)

    report = {
        "users": num_users,
        "K": top_k,
        "global_size": len(global_ordered_items),
        "overlap@K": summarize(overlap_fractions),
        "global_ndcg@K": summarize(global_ndcgs),
        "rbo_to_global": summarize(rbo_values),
        "exposure_share_global": float(exposure_share_global),
        "position_global_share": [float(x) for x in position_global_share],
        "exact_sequence_match_rate": exact_sequence_matches / float(num_users),
        "exact_set_match_rate": exact_set_matches / float(num_users),
    }

    return report

def pretty_print_global(rg: Dict):
    print("\n=== Global bias (optional) ===")
    print(f"# Users: {rg['users']} (K={rg['K']}, global_size={rg['global_size']})\n")

    print("Overlap with global set (fraction of user list in global):")
    for k, v in rg["overlap@K"].items():
        print(f"     {k}: {v:.4f}")

    print("\nGlobal-NDCG@K (rank-aware overlap; 1 = list mirrors global at the top):")
    for k, v in rg["global_ndcg@K"].items():
        print(f"     {k}: {v:.4f}")

    print("\nRBO vs global ordered list (top-heavy; 1 = identical):")
    for k, v in rg["rbo_to_global"].items():
        print(f"     {k}: {v:.4f}")

    print(f"\nExposure share going to global items: {rg['exposure_share_global']:.4f}")
    print(f"Exact sequence match rate (user list == global order): {rg['exact_sequence_match_rate']:.4f}")
    print(f"Exact set match rate     (set(user list) == set(global top-K)): {rg['exact_set_match_rate']:.4f}")

    pos = rg["position_global_share"]
    show = [0,1,2,4,9,19,49,99]  # ranks 1,2,3,5,10,20,50,100 (0-indexed)
    print("\nPosition-wise global share (fraction of users where rank j ∈ global):")
    for j in show:
        if j < len(pos):
            print(f"  rank {j+1:>3}: {pos[j]:.4f}")

# ----------------- CLI -----------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing per-user top-k JSON results")
    parser.add_argument("--k", type=int, default=100, help="Top-K cutoff")
    parser.add_argument("--pair-samples", type=int, default=10000, help="Number of user pairs to sample")
    parser.add_argument("--rbo-p", type=float, default=0.9, help="RBO persistence parameter (0.8–0.98 typical)")
    parser.add_argument("--global-list", default=None, help="File with global top list (JSON array or newline/comma-separated)")
    args = parser.parse_args()

    lists = load_lists(args.dir, K=args.k)
    if not lists:
        print("No user lists found in the specified directory.")
        return

    personalize_report = personalization_overview(lists, args.k, args.pair_samples, args.rbo_p)
    if not personalize_report:
        print("No personalization report generated. Ensure there are enough users and pairs.")
        return

    pretty_print_report(personalize_report)

    if args.global_list:
        global_items = load_global_list(args.global_list, max_n=args.k)
        global_bias_report = global_bias_overview(lists, global_items, top_k=args.k, rbo_p=args.rbo_p)
        if not global_bias_report:
            print("No global bias report generated. Ensure the global list is valid.")
            return
        pretty_print_global(global_bias_report)

if __name__ == "__main__":
    main()