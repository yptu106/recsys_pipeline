import pandas as pd
from collections import defaultdict

def split_repeat_novel(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the positive interactions in the test set into:
        - repeat_df: interactions where the user has seen the streamer in training
        - novel_df: interactions where the user has NOT seen the streamer in training

    Args:
        train_df (pd.DataFrame): Training interactions (must include 'user_id', 'streamer_id', 'label')
        test_df (pd.DataFrame): Testing interactions (must include 'user_id', 'streamer_id', 'label')

    Returns:
        repeat_df (pd.DataFrame): Positive test interactions where streamer was seen in training
        novel_df (pd.DataFrame): Positive test interactions where streamer was NOT seen in training
    """
    # Build user → set of streamers in train set (only positive interactions)
    user_to_train_streamers = defaultdict(set)
    for row in train_df.itertuples():
        if row.label == 1:
            user_to_train_streamers[row.user_id].add(row.streamer_id)

    # Extract positive test interactions
    test_pos_df = test_df[test_df["label"] == 1].copy()

    # Flag each row as repeat or novel
    test_pos_df["is_repeat"] = test_pos_df.apply(
        lambda row: row.streamer_id in user_to_train_streamers[row.user_id],
        axis=1
    )

    # Split
    repeat_df = test_pos_df[test_pos_df["is_repeat"] == True].drop(columns="is_repeat")
    novel_df = test_pos_df[test_pos_df["is_repeat"] == False].drop(columns="is_repeat")

    return repeat_df, novel_df
