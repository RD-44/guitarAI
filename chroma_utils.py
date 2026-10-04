"""
Shared audio -> chroma feature extraction.

A chroma vector has 12 numbers, one per pitch class (C, C#, D, ... B),
representing how much energy is present at each note regardless of
octave or which instrument is playing it. That makes it *the* natural
feature for chord recognition: a chord is just a particular pattern
across these 12 bins (e.g. C major lights up C, E, and G). Every other
script in this project imports extract_chroma() from here so the
recording, training, and live-detection code all agree on exactly how
audio gets turned into numbers.
"""
import numpy as np
import librosa
import os

DATA_DIR = "data"
SR = 22050                        # sample rate we standardize on everywhere
FMIN = librosa.note_to_hz("C2")   # guitar's lowest open string (E2, ~82Hz)
                                   # sits comfortably above this. Librosa's
                                   # chroma_cqt default starts an octave
                                   # lower (C1), which needs longer audio
                                   # windows than we want for a responsive
                                   # real-time detector.


class ChordPredictor:

    def __init__(self):
        self.num_chords = self.count_chords(DATA_DIR)


    def extract_chroma(self, y: np.ndarray, sr: int = SR) -> np.ndarray:
        """Turn a mono audio clip into a single 12-dim chroma vector,
        averaged over time. y should be a 1D float32 array."""
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=FMIN)
        return chroma.mean(axis=1)

    def load_chord_data(self, path: str) -> np.ndarray:
        """Load an audio for all chords and extract chroma vectors."""
        y, sr = librosa.load(path, sr=SR, mono=True)
        return self.extract_chroma(y, sr)

    def count_chords(self, data_dir):
        """Return number of chord labels (directories in data_dir minus 1 for 'N')."""
        if not os.path.exists(data_dir):
            return 0
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return max(0, len(dirs) - 1)

    def predict(k: int, batch: np.ndarray):
        """Predicts chord for each sample in the batch

        Args:
            k (int): Parameter for KNN
            batch (np.ndarray): Batch of inputs (12D chroma feature vectors)
        """
        pass