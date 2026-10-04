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
import warnings
import numpy as np
import librosa

SR = 22050                        # sample rate we standardize on everywhere
FMIN = librosa.note_to_hz("C2")   # guitar's lowest open string (E2, ~82Hz)
                                   # sits comfortably above this. Librosa's
                                   # chroma_cqt default starts an octave
                                   # lower (C1), which needs longer audio
                                   # windows than we want for a responsive
                                   # real-time detector.


def extract_chroma(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Turn a mono audio clip into a single 12-dim chroma vector,
    averaged over time. y should be a 1D float32 array."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_fft=.*too large for input signal")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=FMIN)
    return chroma.mean(axis=1)


def load_and_extract(path: str) -> np.ndarray:
    """Load an audio file from disk and extract its chroma vector."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    return extract_chroma(y, sr)