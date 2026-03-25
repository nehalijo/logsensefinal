"""
main.py
-------
Full pipeline: Parse → Sequence → Detect → Visualize

Usage:
    python main.py                         # runs demo with synthetic data
    python main.py --log data/HDFS.log     # real HDFS log file
    python main.py --log data/HDFS.log --labels data/anomaly_label.csv  # with labels
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.parser import parse_log_file, get_log_summary
from src.sequencer import build_sequences, attach_labels
from src.detector import AnomalyDetector
from src.visualizer import (
    plot_log_level_distribution,
    plot_anomaly_results,
    plot_sequence_heatmap,
)


def run_demo():
    """Run the pipeline on synthetic data so you can verify everything works."""
    print("=" * 60)
    print("  LOG ANOMALY DETECTION — DEMO MODE (synthetic data)")
    print("=" * 60)

    np.random.seed(42)

    # Simulate ~500 normal sequences and 25 anomalous ones
    n_normal, n_anomaly = 500, 25
    normal_seqs  = np.random.choice([0, 1], size=(n_normal, 20), p=[0.97, 0.03])
    anomaly_seqs = np.zeros((n_anomaly, 20), dtype=int)
    for i in range(n_anomaly):
        positions = np.random.choice(20, size=np.random.randint(3, 8), replace=False)
        anomaly_seqs[i, positions] = np.random.choice([2, 3], size=len(positions))

    sequences_arr = np.vstack([normal_seqs, anomaly_seqs])
    block_ids = [f"blk_{i}" for i in range(len(sequences_arr))]
    true_labels = np.array([0] * n_normal + [1] * n_anomaly)

    print(f"\n[Demo] {n_normal} normal + {n_anomaly} anomalous sequences created.")

    # ── Step 3: Detect ────────────────────────────────────────────
    detector = AnomalyDetector(mode="unsupervised")
    detector.train(sequences_arr)
    predictions = detector.predict(sequences_arr)

    detected = predictions.sum()
    print(f"\n[Demo] Detected {detected} anomalies out of {len(predictions)} sequences.")

    # ── Step 4: Visualize ─────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    plot_anomaly_results(block_ids, predictions, labels=true_labels,
                          save_path="outputs/anomaly_results.png")
    plot_sequence_heatmap(sequences_arr, predictions,
                           save_path="outputs/sequence_heatmap.png")

    print("\n✅  Demo complete! Check the outputs/ folder for plots.")


def run_pipeline(log_file: str, label_file: str = None):
    """Run the full pipeline on a real log file."""
    print("=" * 60)
    print("  LOG ANOMALY DETECTION — PIPELINE")
    print("=" * 60)

    # ── Step 1: Parse ─────────────────────────────────────────────
    df = parse_log_file(log_file)
    get_log_summary(df)

    Path("outputs").mkdir(exist_ok=True)
    plot_log_level_distribution(df, save_path="outputs/log_levels.png")

    # ── Step 2: Build sequences ───────────────────────────────────
    seq_data = build_sequences(df, max_len=20)

    if label_file:
        seq_data = attach_labels(seq_data, label_file)
        mode = "supervised"
    else:
        mode = "unsupervised"

    # ── Step 3: Detect ────────────────────────────────────────────
    detector = AnomalyDetector(mode=mode)
    detector.train(
        seq_data["sequences"],
        labels=seq_data.get("labels"),
    )
    predictions = detector.predict(seq_data["sequences"])
    detector.save("models/detector.pkl")

    # ── Step 4: Visualize ─────────────────────────────────────────
    plot_anomaly_results(
        seq_data["block_ids"], predictions,
        labels=seq_data.get("labels"),
        save_path="outputs/anomaly_results.png",
    )
    plot_sequence_heatmap(
        seq_data["sequences"], predictions,
        save_path="outputs/sequence_heatmap.png",
    )

    # ── Step 5: Save results ──────────────────────────────────────
    results_df = pd.DataFrame({
        "BlockId": seq_data["block_ids"],
        "Prediction": ["Anomaly" if p == 1 else "Normal" for p in predictions],
    })
    if "labels" in seq_data:
        results_df["TrueLabel"] = ["Anomaly" if l == 1 else "Normal"
                                    for l in seq_data["labels"]]
    results_df.to_csv("outputs/results.csv", index=False)

    anomaly_count = predictions.sum()
    print(f"\n✅  Done! {anomaly_count} anomalies detected.")
    print("   Outputs saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Anomaly Detection System")
    parser.add_argument("--log",    type=str, help="Path to log file (e.g. data/HDFS.log)")
    parser.add_argument("--labels", type=str, help="Path to label CSV (optional)")
    args = parser.parse_args()

    if args.log:
        run_pipeline(args.log, args.labels)
    else:
        run_demo()
