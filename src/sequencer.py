"""
src/sequencer.py
----------------
Groups parsed log entries into event sequences per Block ID.
Each block's sequence of log levels becomes one "session" for the ML model.

This is the key insight: instead of analyzing individual log lines,
we analyze the SEQUENCE of events for each block — e.g.:
  Block blk_123 → ["INFO", "INFO", "INFO", "ERROR"]  ← anomaly!
  Block blk_456 → ["INFO", "INFO", "INFO", "INFO"]   ← normal
"""

import pandas as pd
import numpy as np
from collections import defaultdict


# Map log levels to numeric event keys
EVENT_MAP = {
    "INFO":    0,
    "WARN":    1,
    "WARNING": 1,
    "ERROR":   2,
    "FATAL":   3,
    "DEBUG":   4,
}


def build_sequences(df: pd.DataFrame, max_len: int = 20) -> dict:
    """
    Group log entries by Block ID into event sequences.

    Args:
        df: Parsed log DataFrame from parser.py
        max_len: Maximum sequence length (pad/truncate to this size)

    Returns:
        dict with:
          - 'sequences': list of fixed-length numeric sequences
          - 'block_ids': corresponding block IDs
          - 'raw_sequences': list of raw level strings (for debugging)
    """
    block_sequences = defaultdict(list)

    for _, row in df.iterrows():
        level = row.get("Level", "INFO").upper()
        event_id = EVENT_MAP.get(level, 0)
        for block_id in row.get("BlockId", []):
            block_sequences[block_id].append(event_id)

    block_ids = []
    sequences = []
    raw_sequences = []

    for block_id, seq in block_sequences.items():
        block_ids.append(block_id)
        raw_sequences.append(seq)
        # Pad with 0 (INFO) or truncate to max_len
        padded = seq[:max_len] + [0] * max(0, max_len - len(seq))
        sequences.append(padded)

    print(f"[Sequencer] Built {len(sequences):,} sequences from {len(df):,} log lines.")
    return {
        "sequences": np.array(sequences),
        "block_ids": block_ids,
        "raw_sequences": raw_sequences,
    }


def attach_labels(sequences: dict, label_file: str) -> dict:
    """
    Attach ground-truth anomaly labels if a label file is available.
    HDFS label file format: 'BlockId,Label' (Label: Normal / Anomaly)

    Args:
        sequences: Output of build_sequences()
        label_file: Path to anomaly_label.csv

    Returns:
        sequences dict with added 'labels' key (1=anomaly, 0=normal)
    """
    labels_df = pd.read_csv(label_file)
    label_map = dict(zip(
        labels_df["BlockId"],
        labels_df["Label"].map({"Normal": 0, "Anomaly": 1})
    ))

    labels = [label_map.get(bid, -1) for bid in sequences["block_ids"]]
    sequences["labels"] = np.array(labels)

    total = len(labels)
    anomalies = sum(1 for l in labels if l == 1)
    print(f"[Sequencer] Labels attached: {anomalies:,} anomalies out of {total:,} sequences "
          f"({anomalies/total*100:.1f}%)")
    return sequences


if __name__ == "__main__":
    # Demo: create a tiny synthetic dataset to verify the module works
    demo_data = {
        "Level": ["INFO", "INFO", "ERROR", "INFO", "INFO", "WARN"],
        "BlockId": [["blk_1"], ["blk_1"], ["blk_1"], ["blk_2"], ["blk_2"], ["blk_2"]],
    }
    df = pd.DataFrame(demo_data)
    result = build_sequences(df, max_len=5)
    for bid, seq in zip(result["block_ids"], result["sequences"]):
        print(f"  {bid}: {seq}")
