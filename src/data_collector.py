"""
Data Collector Module
Records custom sign examples for training
Saves landmarks as training data
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent))

import config
from landmark_extractor import LandmarkExtractor
import json
from datetime import datetime


class DataCollector:
    def __init__(self):
        self.extractor = LandmarkExtractor()
        self.current_sign: Optional[str] = None
        self.recordings: List[List[np.ndarray]] = []
        self._active_sample: List[np.ndarray] = []
        self.frame_count = 0
        self.sample_count = 0

    def start_recording(self, sign_name: str, num_samples: int = config.SAMPLES_PER_SIGN):
        """Call ONCE per sign to initialize. Resets recordings list."""
        self.current_sign = sign_name
        self.recordings = []
        self._active_sample = []
        self.sample_count = 0
        self.frame_count = 0
        print(f"[COLLECTOR] Starting to record '{sign_name}'")
        print(f"[COLLECTOR] Target: {num_samples} samples")

    def begin_sample(self):
        """Call when SPACE is pressed to START a sample. Does NOT reset recordings."""
        self._active_sample = []

    def record_frame(self, frame: np.ndarray) -> bool:
        """Record a frame into the active sample buffer."""
        if self.current_sign is None:
            return False
        landmarks, metadata = self.extractor.extract_landmarks(frame)
        if landmarks is None:
            return False
        self._active_sample.append(landmarks)
        self.frame_count += 1
        return True

    def finish_current_sample(self) -> bool:
        """Call when SPACE is pressed to STOP a sample. Validates and stores it."""
        frames = self._active_sample
        self._active_sample = []

        if len(frames) < config.MIN_FRAMES_PER_SAMPLE:
            print(f"[COLLECTOR] Sample too short ({len(frames)} frames), discarding")
            return False

        self.recordings.append(frames)
        print(f"[COLLECTOR] Sample {len(self.recordings)} recorded ({len(frames)} frames)")
        return True

    def save_recordings(self) -> bool:
        """Save all recordings to disk, appending to existing samples."""
        if not self.current_sign or not self.recordings:
            print("[COLLECTOR] No recordings to save")
            return False
        try:
            sign_dir = config.CUSTOM_DATA_DIR / self.current_sign
            sign_dir.mkdir(parents=True, exist_ok=True)

            existing = list(sign_dir.glob("sample_*.npy"))
            start_index = len(existing)

            for i, recording in enumerate(self.recordings):
                filename = sign_dir / f"sample_{start_index + i:03d}.npy"
                np.save(str(filename), np.array(recording, dtype=np.float32))

            print(f"[COLLECTOR] Saved {len(self.recordings)} samples to {sign_dir}")
            self._update_metadata()
            return True
        except Exception as e:
            print(f"[COLLECTOR] Error saving recordings: {e}")
            return False

    def _update_metadata(self):
        metadata_path = config.CUSTOM_DATA_DIR / "metadata.json"
        try:
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {"signs": []}

            if self.current_sign not in metadata["signs"]:
                metadata["signs"].append(self.current_sign)
                metadata["signs"].sort()

            metadata["last_updated"] = datetime.now().isoformat()
            metadata["total_custom_signs"] = len(metadata["signs"])

            samples_per_sign = {}
            for sign in metadata["signs"]:
                sign_dir = config.CUSTOM_DATA_DIR / sign
                if sign_dir.exists():
                    samples_per_sign[sign] = len(list(sign_dir.glob("sample_*.npy")))
            metadata["samples_per_sign"] = samples_per_sign

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[COLLECTOR] Updated metadata: {metadata['total_custom_signs']} custom signs")
        except Exception as e:
            print(f"[COLLECTOR] Error updating metadata: {e}")

    def get_recordings_count(self) -> int:
        return len(self.recordings)

    def get_total_frames(self) -> int:
        return sum(len(r) for r in self.recordings)

    def reset(self):
        self.current_sign = None
        self.recordings = []
        self._active_sample = []
        self.frame_count = 0
        self.sample_count = 0


def collect_sign_interactive():
    collector = DataCollector()
    cap = cv2.VideoCapture(0)

    print("\n" + "=" * 60)
    print("CUSTOM SIGN DATA COLLECTOR")
    print("=" * 60)

    while True:
        sign_name = input("\nEnter sign name (or 'quit' to exit): ").strip()
        if sign_name.lower() == "quit":
            break
        if not sign_name:
            print("Please enter a sign name")
            continue

        try:
            num_samples = int(
                input(f"Number of samples (default {config.SAMPLES_PER_SIGN}): ")
                or config.SAMPLES_PER_SIGN
            )
        except ValueError:
            num_samples = config.SAMPLES_PER_SIGN

        # ONCE per sign — initialises recordings list
        collector.start_recording(sign_name, num_samples)

        print(f"\nRecording '{sign_name}'...")
        print("Instructions:")
        print("- Position your hand in front of the camera")
        print("- Press SPACE to start recording a sample")
        print("- Press SPACE again to finish the sample")
        print("- Press 'r' to reset current sign")
        print("- Press 'q' to save and move to next sign")
        print("- Press ESC to quit without saving")

        recording_sample = False
        current_sample_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, metadata = collector.extractor.extract_landmarks(frame)

            if recording_sample:
                color = (0, 0, 255)
                status_text = f"RECORDING - Sample {collector.get_recordings_count() + 1}/{num_samples} ({current_sample_frames} frames)"
            else:
                color = (0, 255, 0)
                status_text = f"READY - Sample {collector.get_recordings_count() + 1}/{num_samples}"

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            hand_msg = "Hand detected" if metadata["hands_detected"] else "No hand detected"
            hand_color = (0, 255, 0) if metadata["hands_detected"] else (0, 0, 255)
            cv2.putText(frame, hand_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 1)

            if landmarks is not None:
                frame = collector.extractor.draw_landmarks(frame, landmarks)

            cv2.imshow("Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\nAborted without saving")
                cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord("q"):
                if collector.get_recordings_count() > 0:
                    collector.save_recordings()
                    print(f"Saved {collector.get_recordings_count()} samples")
                else:
                    print("No samples recorded")
                collector.reset()
                break

            elif key == ord("r"):
                collector.reset()
                collector.start_recording(sign_name, num_samples)
                recording_sample = False
                current_sample_frames = 0
                print("Reset - ready for new samples")

            elif key == 32:  # SPACE
                if not recording_sample:
                    collector.begin_sample()          # start fresh sample buffer
                    recording_sample = True
                    current_sample_frames = 0
                    print("Started recording sample")
                else:
                    recording_sample = False
                    collector.finish_current_sample()
                    current_sample_frames = 0
                    if collector.get_recordings_count() >= num_samples:
                        print(f"[COLLECTOR] All {num_samples} samples done!")
                        collector.save_recordings()
                        print(f"Saved {collector.get_recordings_count()} samples")
                        collector.reset()
                        break

            if recording_sample:
                if collector.record_frame(frame):
                    current_sample_frames += 1

                # Auto-stop at max duration
                if current_sample_frames >= config.RECORDING_DURATION * config.EXPECTED_FPS:
                    recording_sample = False
                    collector.finish_current_sample()
                    current_sample_frames = 0
                    print(f"Sample {collector.get_recordings_count()} completed (auto-saved)")
                    if collector.get_recordings_count() >= num_samples:
                        print(f"[COLLECTOR] All {num_samples} samples done!")
                        collector.save_recordings()
                        print(f"Saved {collector.get_recordings_count()} samples")
                        collector.reset()
                        break

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection finished")


if __name__ == "__main__":
    collect_sign_interactive()
