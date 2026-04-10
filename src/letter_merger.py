"""
Letter Merger Module
Converts sequence of sign predictions into coherent words
Implements hold time validation and confidence thresholding
"""

from typing import List, Tuple, Dict, Optional
from collections import Counter
import config
from dataclasses import dataclass, field
from enum import Enum
import time


class LetterState(Enum):
    """States for letter detection"""
    INITIAL = "initial"
    BUILDING = "building"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"


@dataclass
class LetterBuffer:
    """Represents a buffered letter prediction"""
    letter: str
    confidence: float
    frame_count: int
    start_time: float
    predictions: List[Tuple[str, float]] = field(default_factory=list)


class LetterMerger:
    """
    Converts LSTM predictions into words using temporal smoothing.

    Algorithm:
    1. Aggregate predictions over N frames
    2. Apply confidence threshold (>= CONFIDENCE_THRESHOLD)
    3. Group consecutive same-sign predictions
    4. Validate hold time (>= MIN_HOLD_TIME seconds)
    5. Confirm letter once; prevent duplicate confirmation in same hold cycle
    6. Output merged word character by character
    """

    def __init__(self):
        self.current_letter_buffer: Optional[LetterBuffer] = None
        self.confirmed_letters: List[LetterBuffer] = []
        self.prediction_history: List[Tuple[str, float, float]] = []
        self.fps_estimate = config.EXPECTED_FPS
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.min_hold_frames = config.MIN_HOLD_FRAMES
        self.max_hold_frames = config.MAX_HOLD_FRAMES

    def update_fps(self, current_time: float):
        delta = current_time - self.last_frame_time
        if delta > 0:
            self.fps_estimate = 0.8 * self.fps_estimate + 0.2 * (1.0 / delta)
        self.last_frame_time = current_time

    def update_min_hold_frames(self):
        self.min_hold_frames = max(1, int(config.MIN_HOLD_TIME * self.fps_estimate))
        self.max_hold_frames = max(1, int(config.MAX_HOLD_TIME * self.fps_estimate))

    def _last_confirmed_is(self, letter: str) -> bool:
        """Prevent duplicate confirmation of the same held sign."""
        return bool(self.confirmed_letters and self.confirmed_letters[-1].letter == letter)

    def process_prediction(self, predicted_sign: str, confidence: float,
                           all_probs: Dict[str, float]) -> Tuple[str, List[str]]:
        current_time = time.time()
        self.update_fps(current_time)
        self.update_min_hold_frames()
        self.frame_count += 1

        self.prediction_history.append((predicted_sign, confidence, current_time))
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]

        if confidence < config.CONFIDENCE_THRESHOLD:
            current_word = self._check_letter_confirmation()
            return current_word, self._get_confirmed_letters()

        if self.current_letter_buffer is None:
            self.current_letter_buffer = LetterBuffer(
                letter=predicted_sign,
                confidence=confidence,
                frame_count=1,
                start_time=current_time,
                predictions=[(predicted_sign, confidence)]
            )
            return self.get_current_word(), self._get_confirmed_letters()

        if predicted_sign == self.current_letter_buffer.letter:
            buf = self.current_letter_buffer
            buf.frame_count += 1
            buf.confidence = (
                (buf.confidence * (buf.frame_count - 1) + confidence)
                / buf.frame_count
            )
            buf.predictions.append((predicted_sign, confidence))

            hold_time = current_time - buf.start_time

            # FIX: actually confirm the letter when hold criteria are met
            if buf.frame_count >= self.min_hold_frames and hold_time >= config.MIN_HOLD_TIME:
                if not self._last_confirmed_is(predicted_sign):
                    self.confirmed_letters.append(buf)
                    self.current_letter_buffer = None

            return self.get_current_word(), self._get_confirmed_letters()

        buf = self.current_letter_buffer
        hold_time = current_time - buf.start_time

        if (
            buf.frame_count >= self.min_hold_frames
            and hold_time >= config.MIN_HOLD_TIME
            and buf.confidence >= config.CONFIDENCE_THRESHOLD
        ):
            if not self._last_confirmed_is(buf.letter):
                self.confirmed_letters.append(buf)

        self.current_letter_buffer = LetterBuffer(
            letter=predicted_sign,
            confidence=confidence,
            frame_count=1,
            start_time=current_time,
            predictions=[(predicted_sign, confidence)]
        )

        return self.get_current_word(), self._get_confirmed_letters()

    def _check_letter_confirmation(self) -> str:
        if self.current_letter_buffer is None:
            return self.get_current_word()

        buf = self.current_letter_buffer
        hold_time = time.time() - buf.start_time

        if (
            buf.frame_count >= self.min_hold_frames
            and hold_time >= config.MIN_HOLD_TIME
            and buf.confidence >= config.CONFIDENCE_THRESHOLD
        ):
            if not self._last_confirmed_is(buf.letter):
                self.confirmed_letters.append(buf)
            self.current_letter_buffer = None

        return self.get_current_word()

    def _get_confirmed_letters(self) -> List[str]:
        return [lb.letter for lb in self.confirmed_letters]

    def get_current_word(self) -> str:
        return ''.join(lb.letter for lb in self.confirmed_letters)

    def get_buffer_info(self) -> Dict:
        buf = self.current_letter_buffer
        return {
            'confirmed_letters': self._get_confirmed_letters(),
            'current_word': self.get_current_word(),
            'buffered_letter': buf.letter if buf else None,
            'buffer_frame_count': buf.frame_count if buf else 0,
            'buffer_confidence': buf.confidence if buf else 0.0,
            'fps_estimate': self.fps_estimate,
            'total_frames': self.frame_count,
            'min_hold_frames': self.min_hold_frames,
            'max_hold_frames': self.max_hold_frames,
        }

    def get_prediction_stats(self) -> Dict:
        if not self.prediction_history:
            return {}

        recent = self.prediction_history[-config.EXPECTED_FPS:]
        letters = [p[0] for p in recent]
        confidences = [p[1] for p in recent]
        letter_counts = Counter(letters)

        return {
            'recent_predictions': recent[-5:],
            'most_common_letter': letter_counts.most_common(1)[0][0] if letter_counts else "NONE",
            'letter_frequency': dict(letter_counts),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0,
        }

    def reset(self):
        self.current_letter_buffer = None
        self.confirmed_letters = []
        self.prediction_history = []
        self.frame_count = 0
        print("[MERGER] Reset - starting new word")

    def finalize_word(self) -> str:
        if self.current_letter_buffer is not None:
            if self.current_letter_buffer.frame_count >= self.min_hold_frames:
                if not self._last_confirmed_is(self.current_letter_buffer.letter):
                    self.confirmed_letters.append(self.current_letter_buffer)
            self.current_letter_buffer = None

        return self.get_current_word()

    def undo_last_letter(self) -> str:
        if self.confirmed_letters:
            removed = self.confirmed_letters.pop()
            print(f"[MERGER] Removed '{removed.letter}', word: {self.get_current_word()}")

        return self.get_current_word()

    def get_recent_predictions(self, n: int = 10) -> List[Tuple[str, float, float]]:
        return self.prediction_history[-n:]


def test_letter_merger():
    print("[TEST] Starting letter merger test...")

    merger = LetterMerger()

    test_sequence = [
        ('A', 0.44),
        ('A', 0.44),
        ('A', 0.44),
        ('A', 0.44),
        ('A', 0.44),
        ('B', 0.44),
        ('B', 0.44),
        ('B', 0.44),
        ('B', 0.44),
        ('C', 0.44),
        ('C', 0.44),
    ]

    print("[TEST] Processing predictions...")
    for i, (letter, conf) in enumerate(test_sequence):
        word, letters = merger.process_prediction(letter, conf, {})
        info = merger.get_buffer_info()
        print(f"[TEST] Frame {i}: {letter} ({conf:.2f}) -> Word: '{word}' | Buffer: {info['buffered_letter']} ({info['buffer_frame_count']} frames)")
        time.sleep(0.03)

    final_word = merger.finalize_word()
    print(f"[TEST] Final word: '{final_word}'")


if __name__ == "__main__":
    test_letter_merger()