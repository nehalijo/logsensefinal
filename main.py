"""
main.py
-------
Full pipeline: Parse → Sequence → Detect → Visualize

Usage:
    python main.py                              # demo with synthetic data (both models)
    python main.py --log data/HDFS.log          # real HDFS log file
    python main.py --log data/HDFS.log --labels data/anomaly_label.csv  # with labels
    python main.py --model autoencoder          # use only autoencoder
    python main.py --model isolation_forest     # use only isolation forest
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.parser import parse_log_file, get_log_summary
from src.sequencer import build_sequences, attach_labels
from src.detector import AnomalyDetector
from src.autoencoder import LogAutoencoder
from src.visualizer import (
    plot_log_level_distribution,
    plot_anomaly_results,
    plot_sequence_heatmap,
)


def run_demo():
    """Run both models on synthetic data and compare results."""
    print("=" * 60)
    print("  LOG ANOMALY DETECTION — DEMO MODE (synthetic data)")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic data — 500 normal, 25 anomalous sequences
    n_normal, n_anomaly = 500, 25
    normal_seqs = np.random.choice([0, 1], size=(n_normal, 20), p=[0.97, 0.03])
    anomaly_seqs = np.zeros((n_anomaly, 20), dtype=int)
    for i in range(n_anomaly):
        positions = np.random.choice(20, size=np.random.randint(3, 8), replace=False)
        anomaly_seqs[i, positions] = np.random.choice([2, 3], size=len(positions))

    sequences_arr = np.vstack([normal_seqs, anomaly_seqs])
    block_ids = [f"blk_{i}" for i in range(len(sequences_arr))]
    true_labels = np.array([0] * n_normal + [1] * n_anomaly)

    print(f"\n[Demo] {n_normal} normal + {n_anomaly} anomalous sequences created.")

    Path("outputs").mkdir(exist_ok=True)

    # ── Model 1: Isolation Forest ─────────────────────────────────
    print("\n" + "─" * 40)
    print("  MODEL 1 — Isolation Forest")
    print("─" * 40)
    iso = AnomalyDetector(mode="unsupervised")
    iso.train(sequences_arr)
    iso_preds = iso.predict(sequences_arr)
    print(f"[Isolation Forest] Detected {iso_preds.sum()} anomalies out of {len(iso_preds)} sequences.")

    plot_anomaly_results(
        block_ids, iso_preds,
        labels=true_labels,
        save_path="outputs/isolation_forest_results.png"
    )

    # ── Model 2: Autoencoder ──────────────────────────────────────
    print("\n" + "─" * 40)
    print("  MODEL 2 — Autoencoder")
    print("─" * 40)
    ae = LogAutoencoder(sequence_length=20, encoding_dim=8)
    ae.train(sequences_arr, labels=true_labels, epochs=50)
    ae_preds = ae.predict(sequences_arr)
    print(f"[Autoencoder] Detected {ae_preds.sum()} anomalies out of {len(ae_preds)} sequences.")

    plot_anomaly_results(
        block_ids, ae_preds,
        labels=true_labels,
        save_path="outputs/autoencoder_results.png"
    )

    # ── Side by side comparison ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)

    true_anomalies = n_anomaly

    iso_correct = int(np.sum((iso_preds == 1) & (true_labels == 1)))
    iso_false   = int(np.sum((iso_preds == 1) & (true_labels == 0)))

    ae_correct  = int(np.sum((ae_preds == 1) & (true_labels == 1)))
    ae_false    = int(np.sum((ae_preds == 1) & (true_labels == 0)))

    print(f"\n{'Model':<25} {'Detected':>10} {'Correct':>10} {'False Alarms':>14}")
    print("-" * 62)
    print(f"{'Isolation Forest':<25} {iso_preds.sum():>10} {iso_correct:>10} {iso_false:>14}")
    print(f"{'Autoencoder':<25} {ae_preds.sum():>10} {ae_correct:>10} {ae_false:>14}")
    print(f"\nTrue anomalies in dataset: {true_anomalies}")

    # ── Heatmap ───────────────────────────────────────────────────
    plot_sequence_heatmap(
        sequences_arr, ae_preds,
        save_path="outputs/sequence_heatmap.png"
    )

    print("\n✅  Demo complete! Check the outputs/ folder for plots.")
    print("    - isolation_forest_results.png")
    print("    - autoencoder_results.png")
    print("    - sequence_heatmap.png")


def run_pipeline(log_file: str, label_file: str = None, model: str = "both"):
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

    labels = seq_data.get("labels")

    results = {}

    # ── Step 3: Isolation Forest ──────────────────────────────────
    if model in ("both", "isolation_forest"):
        print("\n--- Isolation Forest ---")
        iso = AnomalyDetector(mode="unsupervised")
        iso.train(seq_data["sequences"], labels=labels)
        iso_preds = iso.predict(seq_data["sequences"])
        iso.save("models/isolation_forest.pkl")
        results["isolation_forest"] = iso_preds
        plot_anomaly_results(
            seq_data["block_ids"], iso_preds,
            labels=labels,
            save_path="outputs/isolation_forest_results.png"
        )

    # ── Step 4: Autoencoder ───────────────────────────────────────
    if model in ("both", "autoencoder"):
        print("\n--- Autoencoder ---")
        ae = LogAutoencoder(sequence_length=20, encoding_dim=8)
        ae.train(seq_data["sequences"], labels=labels, epochs=50)
        ae_preds = ae.predict(seq_data["sequences"])
        ae.save("models/autoencoder")
        results["autoencoder"] = ae_preds
        plot_anomaly_results(
            seq_data["block_ids"], ae_preds,
            labels=labels,
            save_path="outputs/autoencoder_results.png"
        )

    # ── Step 5: Heatmap ───────────────────────────────────────────
    main_preds = results.get("autoencoder", results.get("isolation_forest"))
    plot_sequence_heatmap(
        seq_data["sequences"], main_preds,
        save_path="outputs/sequence_heatmap.png"
    )

    # ── Step 6: Save results ──────────────────────────────────────
    results_df = pd.DataFrame({"BlockId": seq_data["block_ids"]})
    for model_name, preds in results.items():
        results_df[model_name] = ["Anomaly" if p == 1 else "Normal" for p in preds]
    if labels is not None:
        results_df["TrueLabel"] = ["Anomaly" if l == 1 else "Normal" for l in labels]
    results_df.to_csv("outputs/results.csv", index=False)

    # ── Comparison table ──────────────────────────────────────────
    if labels is not None and len(results) > 1:
        print("\n" + "=" * 62)
        print("  MODEL COMPARISON")
        print("=" * 62)
        valid = labels != -1
        true = labels[valid]

        print(f"\n{'Model':<25} {'Detected':>10} {'Correct':>10} {'False Alarms':>14}")
        print("-" * 62)
        for model_name, preds in results.items():
            p = preds[valid]
            detected    = int(p.sum())
            correct     = int(np.sum((p == 1) & (true == 1)))
            false_alarms = int(np.sum((p == 1) & (true == 0)))
            print(f"{model_name:<25} {detected:>10} {correct:>10} {false_alarms:>14}")

        total_true = int(np.sum(true == 1))
        print(f"\nTrue anomalies in dataset: {total_true}")

    print("\n✅  Done! Outputs saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Anomaly Detection System")
    parser.add_argument("--log",    type=str, help="Path to log file")
    parser.add_argument("--labels", type=str, help="Path to label CSV")
    parser.add_argument("--model",  type=str, default="both",
                        choices=["both", "autoencoder", "isolation_forest"])
    args = parser.parse_args()

    if args.log:
        run_pipeline(args.log, args.labels, args.model)
    else:
        run_demo()