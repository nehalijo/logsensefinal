"""
src/autoencoder.py
------------------
Autoencoder-based anomaly detection on log event sequences.

How it works:
  - Train ONLY on normal sequences
  - Autoencoder learns to compress and reconstruct normal patterns
  - At prediction time, high reconstruction error = anomaly
  - Normal sequences reconstruct well (low error)
  - Anomalous sequences reconstruct poorly (high error)
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import joblib


class LogAutoencoder:
    """
    Autoencoder for unsupervised log anomaly detection.
    Trains on normal sequences only and detects anomalies
    by measuring reconstruction error.
    """

    def __init__(self, sequence_length: int = 20, encoding_dim: int = 8):
        """
        Args:
            sequence_length: Length of each log event sequence
            encoding_dim: Size of the bottleneck layer
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.threshold = None
        self.scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        Simple autoencoder architecture.

        Structure:
          Input(20)
            → Dense(16)   ← encoder
            → Dense(8)    ← bottleneck (compressed representation)
            → Dense(16)   ← decoder
            → Output(20)

        The bottleneck forces the network to learn the essence
        of normal log patterns — it cannot just copy the input.
        """
        inputs = keras.Input(shape=(self.sequence_length,), name="input")

        # Encoder — compress down
        x = layers.Dense(16, activation="relu", name="encoder_1")(inputs)
        encoded = layers.Dense(self.encoding_dim, activation="relu", name="bottleneck")(x)

        # Decoder — rebuild back up
        x = layers.Dense(16, activation="relu", name="decoder_1")(encoded)
        outputs = layers.Dense(self.sequence_length, activation="sigmoid", name="output")(x)

        model = keras.Model(inputs, outputs, name="log_autoencoder")
        model.compile(
            optimizer="adam",
            loss="mse"
        )
        return model

    def _extract_features(self, sequences: np.ndarray) -> np.ndarray:
        """Normalise sequences to 0-1 range for neural network input."""
        return self.scaler.transform(sequences.astype(float))

    def train(self, sequences: np.ndarray, labels: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train the autoencoder on normal sequences only.

        Args:
            sequences: numpy array of shape (n_samples, sequence_length)
            labels: if provided, trains ONLY on sequences labelled 0 (normal)
            epochs: number of training epochs
            batch_size: number of sequences per training step
        """
        self.scaler.fit(sequences.astype(float))

        if labels is not None:
            normal_sequences = sequences[labels == 0]
            print(f"[Autoencoder] Training on {len(normal_sequences):,} normal sequences only.")
        else:
            normal_sequences = sequences
            print(f"[Autoencoder] Training on {len(normal_sequences):,} sequences (unsupervised).")

        X_train = self._extract_features(normal_sequences)

        self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        # Threshold = mean reconstruction error + 2 standard deviations
        # Anything above this is flagged as anomaly
        reconstructions = self.model.predict(X_train, verbose=0)
        errors = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.threshold = np.mean(errors) + 2 * np.std(errors)
        print(f"\n[Autoencoder] Anomaly threshold set to: {self.threshold:.8f}")
        print(f"[Autoencoder] Error stats — min: {errors.min():.8f} "
              f"max: {errors.max():.8f} mean: {errors.mean():.8f}")

        if labels is not None:
            self._evaluate(sequences, labels)

    def _evaluate(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        """Print evaluation metrics against ground truth labels."""
        predictions = self.predict(sequences)
        valid = labels != -1
        print("\n[Autoencoder] Evaluation on full dataset:")
        print(classification_report(
            labels[valid], predictions[valid],
            target_names=["Normal", "Anomaly"]
        ))

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        Returns 1D array — 1 = anomaly, 0 = normal
        """
        X = self._extract_features(sequences)
        reconstructions = self.model.predict(X, verbose=0)
        errors = np.mean(np.power(X - reconstructions, 2), axis=1)
        return (errors > self.threshold).astype(int)

    def reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Return raw reconstruction errors — higher = more anomalous."""
        X = self._extract_features(sequences)
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.power(X - reconstructions, 2), axis=1)

    def save(self, path: str = "models/autoencoder") -> None:
        """Save model and scaler to disk."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump({
            "scaler":          self.scaler,
            "threshold":       self.threshold,
            "sequence_length": self.sequence_length,
            "encoding_dim":    self.encoding_dim
        }, os.path.join(path, "config.pkl"))
        print(f"[Autoencoder] Model saved to {path}/")

    @classmethod
    def load(cls, path: str = "models/autoencoder") -> "LogAutoencoder":
        """Load a saved autoencoder."""
        config = joblib.load(os.path.join(path, "config.pkl"))
        ae = cls(
            sequence_length=config["sequence_length"],
            encoding_dim=config["encoding_dim"]
        )
        ae.model = keras.models.load_model(os.path.join(path, "model.keras"))
        ae.scaler = config["scaler"]
        ae.threshold = config["threshold"]
        print(f"[Autoencoder] Model loaded from {path}/")
        return ae


if __name__ == "__main__":
    np.random.seed(42)
    normal = np.random.choice([0, 1], size=(400, 20), p=[0.97, 0.03])
    anomalies = np.zeros((20, 20), dtype=int)
    for i in range(20):
        pos = np.random.choice(20, size=np.random.randint(3, 8), replace=False)
        anomalies[i, pos] = np.random.choice([2, 3], size=len(pos))

    X = np.vstack([normal, anomalies])
    labels = np.array([0] * 400 + [1] * 20)

    print("=== Autoencoder Smoke Test ===")
    ae = LogAutoencoder(sequence_length=20, encoding_dim=8)
    ae.train(X, labels=labels, epochs=50)
    preds = ae.predict(X)
    print(f"\nDetected {preds.sum()} anomalies out of {len(preds)} sequences.")