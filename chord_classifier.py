"""
KNN-based chord classifier using chroma vectors.
"""
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from chroma_utils import load_and_extract


class ChordClassifier:
    def __init__(self, data_dir="data", n_neighbors=3):
        self.data_dir = data_dir
        self.n_neighbors = n_neighbors
        self.chord_labels = []
        self.clf = None
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        self.chord_labels = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        X_list = []
        y_list = []

        for idx, label in enumerate(self.chord_labels):
            label_dir = os.path.join(self.data_dir, label)
            wav_files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]

            for wav_file in wav_files:
                wav_path = os.path.join(label_dir, wav_file)
                chroma = load_and_extract(wav_path)
                X_list.append(chroma)
                y_list.append(idx)

        if not X_list:
            raise ValueError("No audio files found in data directory")

        self.X = np.array(X_list)
        self.y = np.array(y_list)

        self.clf = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(self.X)))
        self.clf.fit(self.X, self.y)

    def predict(self, chroma_vectors):
        """
        Predict chord labels for a batch of chroma vectors.

        Args:
            chroma_vectors: np.ndarray of shape (n_samples, 12) or (12,)

        Returns:
            List of chord label strings
        """
        if self.clf is None:
            raise RuntimeError("Classifier not initialized. No training data found.")

        chroma_vectors = np.atleast_2d(chroma_vectors)
        indices = self.clf.predict(chroma_vectors)
        return [self.chord_labels[i] for i in indices]

    def predict_proba(self, chroma_vectors):
        """
        Predict class probabilities for a batch of chroma vectors.

        Args:
            chroma_vectors: np.ndarray of shape (n_samples, 12) or (12,)

        Returns:
            np.ndarray of shape (n_samples, n_classes)
        """
        if self.clf is None:
            raise RuntimeError("Classifier not initialized. No training data found.")

        chroma_vectors = np.atleast_2d(chroma_vectors)
        return self.clf.predict_proba(chroma_vectors)

    def get_chord_labels(self):
        """Return the list of chord labels (index = one-hot encoding)."""
        return self.chord_labels.copy()

    def __len__(self):
        return len(self.X)