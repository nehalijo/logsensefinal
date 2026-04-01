"""
src/detector.py
---------------
ML-based anomaly detection on log event sequences.

Uses two approaches:
  1. Isolation Forest  — unsupervised, no labels needed (good for getting started)
  2. Logistic Regression — supervised, needs labels (better accuracy if labels exist)

Start with Isolation Forest. Switch to Logistic Regression once you have labels.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


class AnomalyDetector:
    """Wraps both unsupervised and supervised anomaly detection."""

    def __init__(self, mode: str = "unsupervised"):
        """
        Args:
            mode: 'unsupervised' (Isolation Forest) or 'supervised' (Logistic Regression)
        """
        assert mode in ("unsupervised", "supervised"), "mode must be 'unsupervised' or 'supervised'"
        self.mode = mode
        self.scaler = StandardScaler()

        if mode == "unsupervised":
            # contamination = estimated fraction of anomalies in your data
            # HDFS dataset has ~2.5% anomalies — adjust for your dataset
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.025,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # handles imbalanced anomaly/normal ratio
                random_state=42,
            )

    def _extract_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Convert raw sequences into a feature matrix.

        Features per sequence:
          - Count of each event type (INFO, WARN, ERROR, FATAL, DEBUG)
          - Sequence length (before padding)
          - Error rate (errors / total events)
          - First event type
          - Last event type
        """
        features = []
        for seq in sequences:
            non_zero_len = max(1, np.count_nonzero(seq) + (seq[0] == 0))  # approx real length
            counts = [np.sum(seq == i) for i in range(5)]  # counts per event type
            error_rate = counts[2] / non_zero_len           # ERROR count / length
            first_event = seq[0]
            nonzero_idx = np.nonzero(seq)[0]
            last_nonpad = seq[nonzero_idx[-1]] if len(nonzero_idx) > 0 else 0
            features.append(counts + [non_zero_len, error_rate, first_event, int(last_nonpad)])
        return np.array(features, dtype=float)

    def train(self, sequences: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Train the model.

        Args:
            sequences: numpy array of shape (n_samples, seq_len)
            labels: 1D array of 0/1 labels (required for supervised mode)
        """
        X = self._extract_features(sequences)
        X_scaled = self.scaler.fit_transform(X)

        if self.mode == "unsupervised":
            print(f"[Detector] Training Isolation Forest on {len(X):,} sequences...")
            self.model.fit(X_scaled)

        else:
            assert labels is not None, "labels required for supervised mode"
            # Filter out unlabeled samples (-1)
            mask = labels != -1
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled[mask], labels[mask], test_size=0.2, random_state=42, stratify=labels[mask]
            )
            print(f"[Detector] Training Logistic Regression — train:{len(X_train):,} test:{len(X_test):,}")
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            print("\n[Detector] Evaluation on test set:")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Returns:
            1D array — 1 = anomaly, 0 = normal
        """
        X = self._extract_features(sequences)
        X_scaled = self.scaler.transform(X)

        if self.mode == "unsupervised":
            raw = self.model.predict(X_scaled)
            # Isolation Forest: -1 = anomaly, 1 = normal → remap to 1/0
            return (raw == -1).astype(int)
        else:
            return self.model.predict(X_scaled)

    def save(self, path: str = "models/detector.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "mode": self.mode}, path)
        print(f"[Detector] Model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/detector.pkl") -> "AnomalyDetector":
        data = joblib.load(path)
        detector = cls(mode=data["mode"])
        detector.model = data["model"]
        detector.scaler = data["scaler"]
        print(f"[Detector] Model loaded from {path}")
        return detector


if __name__ == "__main__":
    # Smoke test with random data
    np.random.seed(42)
    normal_seqs = np.random.choice([0, 1], size=(200, 20), p=[0.95, 0.05])
    anomaly_seqs = np.random.choice([0, 2, 3], size=(10, 20), p=[0.5, 0.3, 0.2])
    X = np.vstack([normal_seqs, anomaly_seqs])

    detector = AnomalyDetector(mode="unsupervised")
    detector.train(X)
    preds = detector.predict(X)
    print(f"\nDetected {preds.sum()} anomalies out of {len(preds)} sequences.")
