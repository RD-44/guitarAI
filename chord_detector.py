"""
Chord detection with windowing, KNN prediction, and majority vote smoothing.
"""
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from chord_classifier import ChordClassifier
from chroma_utils import extract_chroma


class ChordDetector:
    def __init__(
        self,
        classifier: ChordClassifier,
        window_seconds: float = 1.0,
        hop_seconds: float = 0.2,
        vote_window: int = 5,
        confidence_threshold: float = 0.5,
        sample_rate: int = 22050,
    ):
        self.classifier = classifier
        self.window_samples = int(window_seconds * sample_rate)
        self.hop_samples = int(hop_seconds * sample_rate)
        self.vote_window = vote_window
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate

        self.prediction_history: List[Tuple[str, float, float]] = []

    def process_window(self, audio_window: np.ndarray) -> Tuple[str, float]:
        if len(audio_window) != self.window_samples:
            raise ValueError(
                f"Expected window of {self.window_samples} samples, got {len(audio_window)}"
            )
        chroma = extract_chroma(audio_window)
        probs = self.classifier.predict_proba(chroma[None, :])[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        label = self.classifier.get_chord_labels()[idx]
        if confidence < self.confidence_threshold:
            label = "N"
        return label, confidence

    def update(self, label: str, confidence: float) -> str:
        now = time.time()
        self.prediction_history.append((label, confidence, now))
        if len(self.prediction_history) > self.vote_window:
            self.prediction_history.pop(0)
        votes = Counter(l for l, _, _ in self.prediction_history)
        return votes.most_common(1)[0][0]

    def get_stats(self) -> Dict[str, Any]:
        if not self.prediction_history:
            return {"history_len": 0, "last_label": None, "last_confidence": None}
        return {
            "history_len": len(self.prediction_history),
            "last_label": self.prediction_history[-1][0],
            "last_confidence": self.prediction_history[-1][1],
        }

    def reset(self):
        self.prediction_history.clear()