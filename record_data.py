"""
Record labeled guitar chord clips from your microphone.

Run this, type the name of the chord you're about to play (e.g. "C",
"Am", "G7" -- any labels you want), strum it when prompted, and it
saves a short .wav into data/<label>/NNN.wav. Aim for 15-30 takes per
chord, varying strum strength, pick vs. fingers, and voicing if you
know more than one shape for it. Also record a handful of clips under
a label like "N" (no chord) -- silence, muted strings, string noise --
so your detector learns to recognize "nothing is being played" too.
"""
import os
import time
import sounddevice as sd
import soundfile as sf

DATA_DIR = "data"
SR = 22050
CLIP_SECONDS = 1.5


def record_clip(seconds=CLIP_SECONDS, sr=SR):
    print(f"  recording for {seconds}s...", end="", flush=True)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print(" done")
    return audio.flatten()


def next_index(label_dir):
    existing = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
    return len(existing) + 1


def main():
    print("Guitar chord data recorder")
    print("Type a chord label and hit enter to record a clip for it.")
    print("Type 'q' to quit.\n")

    while True:
        label = input("Chord label (e.g. C, Am, G7, N=no chord): ").strip()
        if label.lower() == "q":
            break
        if not label:
            continue

        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        print(f"  get ready to play '{label}'...")
        for i in range(3, 0, -1):
            print(f"  {i}...", end=" ", flush=True)
            time.sleep(1)
        print("Go!")
        audio = record_clip()

        idx = next_index(label_dir)
        out_path = os.path.join(label_dir, f"{idx:03d}.wav")
        sf.write(out_path, audio, SR)
        print(f"  saved -> {out_path}\n")

    print("Done. Your dataset is in ./data/<label>/*.wav")


if __name__ == "__main__":
    main()
