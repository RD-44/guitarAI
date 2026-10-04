"""
Live chord detection from microphone.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from audio_stream import AudioStream
from chord_classifier import ChordClassifier
from chord_detector import ChordDetector


DEFAULTS = {
    "audio": {
        "sample_rate": 22050,
        "blocksize": 1024,
        "buffer_seconds": 2.0,
    },
    "detector": {
        "window_seconds": 1.0,
        "hop_seconds": 0.2,
        "vote_window": 5,
        "confidence_threshold": 0.5,
    },
    "classifier": {
        "n_neighbors": 3,
        "data_dir": "data",
    },
    "display": {
        "update_interval": 0.1,
        "show_confidence": True,
    },
    "logging": {
        "enabled": True,
        "path": "logs/detect_{timestamp}.jsonl",
        "level": "INFO",
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return DEFAULTS.copy()
    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}
    return deep_merge(DEFAULTS, user_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live guitar chord detection")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--window", type=float, help="Window size in seconds")
    parser.add_argument("--hop", type=float, help="Hop size in seconds")
    parser.add_argument("--vote-window", type=int, help="Majority vote window size")
    parser.add_argument("--threshold", type=float, help="Confidence threshold")
    parser.add_argument("--k", type=int, dest="n_neighbors", help="KNN neighbors")
    parser.add_argument("--data-dir", help="Data directory")
    parser.add_argument("--blocksize", type=int, help="Audio blocksize")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--no-log", action="store_true", help="Disable logging")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    return parser.parse_args()


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    if args.window is not None:
        config["detector"]["window_seconds"] = args.window
    if args.hop is not None:
        config["detector"]["hop_seconds"] = args.hop
    if args.vote_window is not None:
        config["detector"]["vote_window"] = args.vote_window
    if args.threshold is not None:
        config["detector"]["confidence_threshold"] = args.threshold
    if args.n_neighbors is not None:
        config["classifier"]["n_neighbors"] = args.n_neighbors
    if args.data_dir is not None:
        config["classifier"]["data_dir"] = args.data_dir
    if args.blocksize is not None:
        config["audio"]["blocksize"] = args.blocksize
    if args.log_file is not None:
        config["logging"]["path"] = args.log_file
    if args.no_log:
        config["logging"]["enabled"] = False
    return config


class JSONLinesHandler(logging.Handler):
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "a")

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.file.write(msg + "\n")
            self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self.file.close()
        super().close()


def setup_logging(log_config: dict) -> logging.Logger:
    logger = logging.getLogger("live_detect")
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    if log_config.get("enabled", True):
        path_template = log_config.get("path", "logs/detect_{timestamp}.jsonl")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = path_template.format(timestamp=timestamp)
        json_handler = JSONLinesHandler(filepath)
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(json_handler)

    return logger


def main():
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    config = load_config(args.config)
    config = merge_config(config, args)

    logger = setup_logging(config["logging"])

    data_dir = config["classifier"]["data_dir"]
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        print(f"Error: Data directory not found: {data_dir}")
        print("Run 'python record_data.py' to record training data first.")
        sys.exit(1)

    try:
        clf = ChordClassifier(
            data_dir=data_dir,
            n_neighbors=config["classifier"]["n_neighbors"],
        )
    except ValueError as e:
        logger.error(f"Failed to load classifier: {e}")
        print(f"Error: {e}")
        sys.exit(1)

    logger.info({
        "event": "classifier_loaded",
        "n_chords": len(clf),
        "labels": clf.get_chord_labels(),
        "n_neighbors": config["classifier"]["n_neighbors"],
    })

    detector = ChordDetector(
        classifier=clf,
        window_seconds=config["detector"]["window_seconds"],
        hop_seconds=config["detector"]["hop_seconds"],
        vote_window=config["detector"]["vote_window"],
        confidence_threshold=config["detector"]["confidence_threshold"],
        sample_rate=config["audio"]["sample_rate"],
    )

    stream = AudioStream(
        sample_rate=config["audio"]["sample_rate"],
        blocksize=config["audio"]["blocksize"],
        buffer_seconds=config["audio"]["buffer_seconds"],
    )
    stream.start()

    logger.info({
        "event": "stream_started",
        "sample_rate": config["audio"]["sample_rate"],
        "blocksize": config["audio"]["blocksize"],
        "buffer_seconds": config["audio"]["buffer_seconds"],
        "window_seconds": config["detector"]["window_seconds"],
        "hop_seconds": config["detector"]["hop_seconds"],
    })

    print("Listening... Press Ctrl+C to stop")
    print(f"Chords: {', '.join(clf.get_chord_labels())}")
    print("-" * 40)

    last_window_end = 0
    display_interval = config["display"]["update_interval"]
    last_display = 0
    show_confidence = config["display"]["show_confidence"]

    try:
        while True:
            target_end = last_window_end + detector.hop_samples
            window = stream.get_window_at(target_end, detector.window_samples)

            if window is not None:
                label, confidence = detector.process_window(window)
                smoothed = detector.update(label, confidence)
                last_window_end = target_end

                logger.info({
                    "event": "prediction",
                    "raw_label": label,
                    "raw_confidence": round(confidence, 4),
                    "smoothed_label": smoothed,
                    "timestamp": time.time(),
                })

                now = time.time()
                if now - last_display >= display_interval:
                    conf_str = f" ({confidence:.2f})" if show_confidence else ""
                    print(f"\r{smoothed}{conf_str}", end="", flush=True)
                    last_display = now
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        logger.info({"event": "stream_stopped"})
        print("\nStopped.")


if __name__ == "__main__":
    main()