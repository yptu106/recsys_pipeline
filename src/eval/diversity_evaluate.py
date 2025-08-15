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
    Get upt to m unique pairs of user IDs to compute pairwise metrics (Jaccard, RBO, etc.)
    without having to iterate over all `n choose 2` pairs.

    Args:
        uids (List[int]): List of user IDs.
        m (int): Number of pairs to sample.

    Returns:
        List[Tuple[int, int]]: List of sampled user pairs.
    """
    if len(uids) < 2: 
        return []
    seen, out = set(), []
    max_pairs = (len(uids) * (len(uids) - 1)) // 2 # n choose 2
    target = min(m, max_pairs)
    while len(out) < target:
        i, j = random.sample(uids, 2)            # pick two distinct users
        a, b = (i, j) if i < j else (j, i)       # canonical order -> (min, max)
        if (a, b) not in seen:
            seen.add((a, b)); out.append((a, b)) # ensure uniqueness
    return out

# ----------------- pairwise similarity -----------------

def jaccard(a: List[int], b: List[int]) -> float:
    """
    Calculate Jaccard similarity between two lists.
    Jaccard similarity is defined as the size of the intersection divided by the size of the union.
    Jaccard(a, b) = |A ∩ B| / |A ∪ B|

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
    RBO(list_a, list_b, p) = (1 - p) * Σ (Ad) * p^(depth - 1)
    Ad (Agreement at depth) = |{items in common at depth}| / depth

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

def exposure_stats(user_top_k_lists: Dict[int, List[int]]):
    """
    Calculate exposure statistics.

    Args:
        lists (Dict[int, List[int]]): Dictionary mapping user IDs to lists of items.

    Returns:
        Dict[str, float]: Dictionary containing exposure statistics:
            - coverage_items: Number of unique items shown.
                - Range: 1 .. N (where N is the number of unique items across all users).
                - Interpretation: Higher values indicate more diverse item exposure.
            - gini: Gini coefficient for item exposure.
                - Inequality of the exposure distribution
                    - p_i = exposure of item i / total exposure
                - Range: 0.0 .. 1.0
                    - 0.0: Perfect equality (all items equally exposed).
                    - 1.0: All exposure on a single item (perfect inequality).
                - Interpretation: Lower values indicate more equal exposure across items.
                    - ~0.2 - 0.3: Good diversity.
                    - ~0.5: moderate concentration.
                    - >0.7: high concentration on a few items.
            - hhi: Herfindahl-Hirschman Index for item exposure.
                - HHI = sum(p_i^2) for all items i (sum of squared shared exposure shares)
                - Range: 1/N .. 1.0 where N is the number of items with non-zero exposure.
                - Interpretation: lower values indicate more diverse exposure.
            - entropy: Normalized entropy of item exposure.
                - H = -sum(p_i * log(p_i)) for all items i, where p_i is the exposure share of item i.
                    - then normalized by log(N) where N is the number of items with non-zero exposure.
                - Range: 0.0 .. 1.0
                    - 0.0: All exposure on a single item (perfect concentration).
                    - 1.0: All items equally exposed (perfect diversity).
                - Interpretation: Higher values indicate more diverse exposure.
    """
    # count how many times each item appears across all users' lists
    item_exposure_counts = Counter(item_id for items in user_top_k_lists.values() for item_id in items)
    total_impressions = sum(item_exposure_counts.values())
    if total_impressions == 0:
        return {"coverage_items": 0, "gini": 0.0, "hhi": 0.0, "entropy": 0.0}

    # sorted array of exposure counts per item
    exposure_counts = np.array(list(item_exposure_counts.values()), dtype=np.float64)
    exposure_counts.sort()
    num_items_exposed = exposure_counts.size

    # gini coefficient
    ranks = np.arange(1, num_items_exposed + 1)
    gini_numerator = np.sum((2 * ranks - num_items_exposed - 1) * exposure_counts)
    gini_denominator = num_items_exposed * np.sum(exposure_counts)
    gini = gini_numerator / gini_denominator if gini_denominator > 0 else 0.0

    # HHI and entropy
    exposure_shares = exposure_counts / float(total_impressions)
    hhi = float(np.sum(exposure_shares ** 2))

    # normalized Shannon entropy (0..1), normalized by log(#exposed items)
    shannon_entropy = -float(np.sum(exposure_shares * np.log(exposure_shares + 1e-12)))
    normalized_entropy = shannon_entropy / log(num_items_exposed) if num_items_exposed > 1 else 0.0

    # coverage is the number of unique items shown
    coverage_items = int(len(item_exposure_counts))

    return {
        "coverage_items": coverage_items, 
        "gini": float(gini), 
        "hhi": hhi, 
        "entropy": normalized_entropy
    }

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
    Calculate the fraction of users whose top-K exact list occurs more than once. 
    - Interpretation: detects broader duplication, even if the mode list isn't huge. 
        - e.g., if many users get the same top-K list, this will be high.
    - Sensitivity: looser than mode_list_share

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
    - mode_list: the most frequently occurring list of items across all users.
    - Interpretation: Higher values indicate that many users have the same list, which may suggest low diversity.
        - bug, cold-start fallback, or an overly strong global prior.
    - Sensitivity: very strict, only fires when many users get the exact same sequence.

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
        # Duplication diagnostics (check are we dumping identical lists to lots of users?)
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
    Compute Discounted Cumulative Gain (DCG) at a cutoff K.

    Parameters
    ----------
    relevance_labels : np.ndarray
        A 1D array of graded or binary relevance values aligned with the ranked list.
        Higher values represent more relevant items. Example for binary relevance:
        1.0 if the item is considered a "hit", 0.0 otherwise.
    top_k : int
        The cutoff rank K at which to compute DCG (only the first K positions are used).

    Returns
    -------
    float
        The DCG value at K (non-negative). Larger is better.

    Notes
    -----
    - DCG discounts lower-ranked hits by log2 of their rank position, emphasizing
      early ranks where user attention is typically higher.
    - This implementation uses the common formulation:
        DCG@K = sum_{i=1..K} (relevance_i / log2(i + 1))
      where rank i is 1-indexed (i=1 is the top item).
    """
    # Consider only the first K labels (if the list is shorter, this slices to its length)
    relevance_labels = relevance_labels[:top_k]

    # If there are no items, the gain is zero
    if relevance_labels.size == 0:
        return 0.0

    # For positions 1..len(relevance_labels), compute 1 / log2(rank + 1)
    # Example: rank 1 weight = 1/log2(2) = 1.0, rank 2 weight ≈ 0.6309, etc.
    discounts = 1.0 / np.log2(np.arange(2, relevance_labels.size + 2))

    # DCG is the sum of (relevance * discount) over positions
    return float((relevance_labels * discounts).sum())


def global_ndcg_at_k(user_ranked_items: List[int], global_popular_item_set: set[int], top_k: int) -> float:
    """
    Compute NDCG@K where "relevance" is membership in a given global popularity set.

    This measures, in a rank-aware way, how closely a user's ranked list aligns
    with a "global popular" reference set (e.g., monthly Top-100). Items inside
    the reference set are treated as relevance = 1; items outside are relevance = 0.

    Parameters
    ----------
    user_ranked_items : List[int]
        The user's ranked list of item identifiers (highest rank first).
    global_popular_item_set : Set[int]
        The set of reference-popular item identifiers (e.g., monthly Top-100).
    top_k : int
        The cutoff rank K at which to compute NDCG (only the first K positions are used).

    Returns
    -------
    float
        NDCG value in [0, 1]. Higher means the user's top ranks contain more items
        from the global set and/or place them higher.

    Steps
    -----
    1) Build a binary relevance vector of length <= K for the user's list:
       relevance[i] = 1.0 if user_ranked_items[i] ∈ global_popular_item_set else 0.0
    2) Compute DCG@K from that vector (rewarding earlier positions more).
    3) Compute the ideal DCG (IDCG) for the best possible ordering under the same constraints:
       this is the DCG of a vector of ones of length min(K, |global set|).
    4) Return DCG / IDCG (or 0.0 if IDCG = 0).

    Edge Cases
    ----------
    - If the global set is empty, the ideal DCG is 0 → return 0.0.
    - If the user list is shorter than K, only the available items are considered.
    """
    # Build a binary relevance vector for the first K user-ranked items
    relevance_labels = np.array(
        [1.0 if item_id in global_popular_item_set else 0.0 for item_id in user_ranked_items[:top_k]],
        dtype=np.float32,
    )

    # Compute DCG for the user's ranked items
    dcg = dcg_at_k(relevance_labels, top_k)

    # Compute the ideal DCG for the best possible ordering:
    # as many ones as possible up to K (bounded by the size of the global set)
    ideal_size = min(top_k, len(global_popular_item_set))
    ideal_relevance = np.ones((ideal_size,), dtype=np.float32)
    ideal_dcg = dcg_at_k(ideal_relevance, ideal_size)

    # normalize DCG by IDCG
    # if ideal_dcg is 0, return 0.0 to avoid division by zero
    # otherwise, return the normalized DCG
    return 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg


def global_bias_overview(user_top_k_lists: Dict[int, List[int]], global_ordered_items: List[int], top_k: int, rbo_p: float):
    """
    - overlap@K: how much of the user's top-K is in the global list
        - High: many items shown are in the global list.
    - global_ndcg@K: are popular items placed at the top?
        - treat items in the global list as relevance=1, and compute nDCG@K
        - High: global items dominate early rank positions.
        - if overlap is high but global_ndcg is low, you're putting global items mostly lower in the list.
        - if global_ndcg is noticeably higher than overlap, global items were concentrated at the very top
    - rbo_to_global: how close is the user list to the exact global order?
        - High: the user list is very similar to the global order.
    - exposure_share_global: how much traffic goes to global items overall
        - across all users and ranks, share of impressions that are in the global list
        - complements (1)-(3), which report per-user metrics.
    - position_global_share: at each rank, how often is the item popular?
        - for position j, fraction of users whose item at j is in the global list
        - if ranks 1-5 have very high shares, your top slots are dominated by the global head.
    - exact_sequence_match_rate: are we literally dumping the global list?
        - should be ~0; any spike indicates a fallback or bug.
    - exact_set_match_rate: are we dumping the global top-K set?
        - catches cases where you shuffle the global list but keep the same items.
    """
    users = list(user_top_k_lists.keys())

    global_ordered_items = global_ordered_items[:top_k]
    global_item_set = set(global_ordered_items)

    overlap_fractions = [] # how much of the user's top-K is in the global list
    global_ndcgs = [] # are popular items placed at the top?
    rbo_values = [] # how close is the user list to the exact global order?

    position_hits = np.zeros(top_k, dtype=np.int64)
    total_global_exposures = 0

    exact_sequence_matches = 0
    exact_set_matches = 0
    global_sequence_tuple = tuple(global_ordered_items)
    global_set_top_k = set(global_ordered_items[:top_k])

    for user_id in tqdm(users, desc="Calculating global bias metrics"):
        ranked_items = user_top_k_lists[user_id][:top_k]

        # fraction of the user's top-K that are in the global list
        # |L_u ∩ L_g| / K
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

    # evaluate whether the user lists are biased towards the global list
    # e.g., if many users get the top-K list that is very similar to the global list (monthly top streamers)
    if args.global_list:
        global_items = load_global_list(args.global_list, max_n=args.k)
        global_bias_report = global_bias_overview(lists, global_items, top_k=args.k, rbo_p=args.rbo_p)
        if not global_bias_report:
            print("No global bias report generated. Ensure the global list is valid.")
            return
        pretty_print_global(global_bias_report)

if __name__ == "__main__":
    main()