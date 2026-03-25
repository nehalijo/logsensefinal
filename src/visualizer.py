"""
src/visualizer.py
-----------------
Visualize log patterns and detected anomalies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

LEVEL_NAMES = {0: "INFO", 1: "WARN", 2: "ERROR", 3: "FATAL", 4: "DEBUG"}
LEVEL_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336", 3: "#9C27B0", 4: "#2196F3"}


def plot_log_level_distribution(df: pd.DataFrame, save_path: str = None):
    """Bar chart of log level counts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Level"].value_counts()
    colors = [LEVEL_COLORS.get({"INFO": 0, "WARN": 1, "WARNING": 1,
                                 "ERROR": 2, "FATAL": 3, "DEBUG": 4}.get(l, 0), "#999")
              for l in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Log Level Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Log Level")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved: {save_path}")
    plt.show()


def plot_anomaly_results(block_ids: list, predictions: np.ndarray,
                          labels: np.ndarray = None, save_path: str = None):
    """
    Scatter plot of sequences — normal vs anomaly.
    If ground-truth labels are provided, also shows false positives/negatives.
    """
    n = len(predictions)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))

    if labels is not None:
        colors = []
        for pred, true in zip(predictions, labels):
            if true == -1:
                colors.append("#CCCCCC")    # unlabeled
            elif pred == 1 and true == 1:
                colors.append("#F44336")    # True Positive (detected anomaly)
            elif pred == 0 and true == 0:
                colors.append("#4CAF50")    # True Negative (normal, correct)
            elif pred == 1 and true == 0:
                colors.append("#FF9800")    # False Positive
            else:
                colors.append("#9C27B0")    # False Negative (missed anomaly!)

        patches = [
            mpatches.Patch(color="#4CAF50", label="True Normal"),
            mpatches.Patch(color="#F44336", label="True Anomaly (detected)"),
            mpatches.Patch(color="#FF9800", label="False Positive"),
            mpatches.Patch(color="#9C27B0", label="False Negative (missed)"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=9)
    else:
        colors = ["#F44336" if p == 1 else "#4CAF50" for p in predictions]
        patches = [
            mpatches.Patch(color="#4CAF50", label="Normal"),
            mpatches.Patch(color="#F44336", label="Anomaly Detected"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=9)

    ax.scatter(x, predictions, c=colors, s=10, alpha=0.7)
    ax.set_title(f"Anomaly Detection Results — {predictions.sum()} anomalies in {n} sequences",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("Sequence Index")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved: {save_path}")
    plt.show()


def plot_sequence_heatmap(sequences: np.ndarray, predictions: np.ndarray,
                           n_samples: int = 30, save_path: str = None):
    """
    Heatmap showing event sequences — anomalies highlighted.
    Shows a sample of normal and anomalous sequences side by side.
    """
    anomaly_idx = np.where(predictions == 1)[0][:n_samples // 2]
    normal_idx = np.where(predictions == 0)[0][:n_samples // 2]
    selected = np.concatenate([anomaly_idx, normal_idx])
    labels_plot = ["⚠ ANOMALY" if predictions[i] == 1 else "✓ normal" for i in selected]

    fig, ax = plt.subplots(figsize=(14, max(6, len(selected) * 0.3)))
    sns.heatmap(
        sequences[selected],
        ax=ax,
        cmap="YlOrRd",
        cbar_kws={"label": "Event Type (0=INFO … 3=FATAL)"},
        linewidths=0.1,
        linecolor="#eee",
        yticklabels=labels_plot,
    )
    ax.set_title("Event Sequence Heatmap (anomalies vs normals)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Step in Sequence")
    ax.set_ylabel("Block Sequence")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved: {save_path}")
    plt.show()
