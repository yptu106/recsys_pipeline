import argparse
import pathlib
import pandas as pd

DEFAULT_K = 100  # Default number of candidates to retrieve

def rank_user(
    user_id: int, 
    pop_streamers_list: list,
    topk: int = DEFAULT_K,
):
    # Get the top K popular streamers for the user
    top_streamers = pop_streamers_list[:topk]

    # Create a DataFrame to save results
    results_df = pd.DataFrame({
        "user_id": user_id,
        "streamer_id": top_streamers,
        "rank": range(1, len(top_streamers) + 1)
    })

    return results_df