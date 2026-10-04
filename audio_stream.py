"""
Audio stream with ring buffer for continuous microphone capture.
"""
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

AudioBuffer = Optional[np.ndarray]


class AudioStream:
    def __init__(
        self,
        sample_rate: int = 22050,
        blocksize: int = 1024,
        buffer_seconds: float = 2.0,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.buffer_seconds = buffer_seconds
        self.device = device

        self.buffer_size = int(sample_rate * buffer_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.lock = threading.Lock()
        self.stream: Optional[sd.InputStream] = None
        self.start_time = 0.0
        self.total_samples = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        samples = indata[:, 0]
        n = len(samples)
        with self.lock:
            end = self.write_pos + n
            if end <= self.buffer_size:
                self.buffer[self.write_pos:end] = samples
            else:
                first_part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = samples[:first_part]
                self.buffer[:end - self.buffer_size] = samples[first_part:]
            self.write_pos = end % self.buffer_size
            self.total_samples += n

    def start(self):
        if self.stream is not None:
            return
        self.start_time = time.time()
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_latest_window(self, window_samples: int) -> AudioBuffer:
        with self.lock:
            available = self.total_samples
            if available < window_samples:
                return None
            end = self.write_pos
            start = (end - window_samples) % self.buffer_size
            if start < end:
                return self.buffer[start:end].copy()
            else:
                return np.concatenate([self.buffer[start:], self.buffer[:end]])

    def get_window_at(self, target_end_sample: int, window_samples: int) -> AudioBuffer:
        with self.lock:
            if self.total_samples < target_end_sample:
                return None
            if self.total_samples - target_end_sample > self.buffer_size:
                return None
            end = target_end_sample % self.buffer_size
            start = (target_end_sample - window_samples) % self.buffer_size
            if start < end:
                return self.buffer[start:end].copy()
            else:
                return np.concatenate([self.buffer[start:], self.buffer[:end]])

    def get_available_samples(self) -> int:
        with self.lock:
            return self.total_samples


def main():
    stream = AudioStream()
    stream.start()
    print("Recording... press Ctrl+C to stop")
    try:
        while True:
            window = stream.get_latest_window(22050)
            if window is not None:
                rms = np.sqrt(np.mean(window**2))
                print(f"\rRMS: {rms:.6f}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()