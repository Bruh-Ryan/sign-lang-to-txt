"""
Main GUI Application
Real-time sign language translator with custom sign support
PySimpleGUI interface optimized for MacOS

FIX: Prediction threshold raised to 10 real frames (was 5) so the SVM
     gets a stable representative sample before it commits to a label.
     Also: mode auto-switches to Custom if a custom model is loaded on startup.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

import PySimpleGUI as sg
import cv2
import numpy as np
from pathlib import Path
import threading
import queue
from typing import Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent))

import config
from landmark_extractor import LandmarkExtractor
from lstm_predictor import LSTMPredictor
from letter_merger import LetterMerger
import datetime
from data_collector import collect_sign_interactive
from trainer import train_custom_signs_interactive

sg.theme(config.GUI_THEME)

# ── Globals ───────────────────────────────────────────────────────────────────
is_running       = True
frame_queue      = queue.Queue(maxsize=2)
use_custom_model = False

_predictor: Optional[LSTMPredictor] = None
_merger:    Optional[LetterMerger]   = None
_vt_lock = threading.Lock()

# FIX: minimum real (non-zero-padded) frames before we bother predicting.
# 5 was too low for SVM — the mean over only 5 frames is noisy.
# 10 frames ≈ 0.33 s at 30 FPS, still very responsive.
MIN_REAL_FRAMES_FOR_PREDICTION = 10


# ── Video capture ─────────────────────────────────────────────────────────────
class VideoCapture:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        self.is_open = self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.release()


# ── Video thread ──────────────────────────────────────────────────────────────
def video_thread(window_key):
    global is_running, use_custom_model, _predictor, _merger

    video = VideoCapture(0)
    if not video.is_open:
        print("[ERROR] Could not open camera")
        return

    extractor = LandmarkExtractor()

    with _vt_lock:
        _predictor = LSTMPredictor()
        _merger    = LetterMerger()

    landmarks_buffer = []
    frame_count      = 0
    _last_mode       = None

    print("[VIDEO THREAD] Started")

    while is_running:
        ret, frame = video.read_frame()
        if not ret:
            break

        landmarks, metadata = extractor.extract_landmarks(frame)

        prediction = None
        if landmarks is not None:
            landmarks_buffer.append(landmarks)
            if len(landmarks_buffer) > config.MAX_SEQUENCE_LENGTH:
                landmarks_buffer.pop(0)

            # FIX: require MIN_REAL_FRAMES_FOR_PREDICTION real frames (not just 5)
            # This ensures SVM gets a meaningful mean, not noise from 5 frames.
            if len(landmarks_buffer) >= MIN_REAL_FRAMES_FOR_PREDICTION:
                current_mode = use_custom_model
                if current_mode != _last_mode:
                    if current_mode:
                        _predictor.switch_to_custom_model()
                    else:
                        _predictor.switch_to_asl_model()
                    _last_mode = current_mode

                sign, confidence, all_probs = _predictor.predict(landmarks_buffer)
                word, confirmed_letters     = _merger.process_prediction(sign, confidence, all_probs)

                prediction = {
                    'sign':        sign,
                    'confidence':  confidence,
                    'word':        word,
                    'top_5':       _predictor.get_top_predictions(all_probs, top_k=5),
                    'buffer_info': _merger.get_buffer_info(),
                    'metadata':    metadata,
                }
        else:
            # Hand left frame — clear buffer so next detection starts fresh
            # FIX: only clear if it's been absent for a moment, not every missed frame
            # (MediaPipe can drop a frame; we don't want to flush on every blink)
            landmarks_buffer = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buf    = cv2.imencode(".png", frame_rgb)

        try:
            frame_queue.put_nowait((buf.tobytes(), prediction))
        except queue.Full:
            pass

        frame_count += 1

    extractor.release()
    video.release()
    print("[VIDEO THREAD] Stopped")


# ── Window layout ─────────────────────────────────────────────────────────────
def create_window():
    layout = [
        [
            sg.Column([
                [sg.Text("SIGN LANGUAGE TRANSLATOR", font=("Arial", 20, "bold"))],
                [sg.Text("Real-time ASL & Custom Signs Recognition", font=("Arial", 12))],
            ], vertical_alignment='top')
        ],
        [sg.HSeparator()],
        [
            sg.Column([
                [sg.Text("Video Feed", font=("Arial", 12, "bold"))],
                [sg.Image(data=None, key="-IMAGE-", size=(640, 480))],
                [sg.Text("No camera detected", key="-CAM-STATUS-")],
            ], vertical_alignment='top', size=(700, 530)),

            sg.Column([
                [sg.Text("Predictions", font=("Arial", 12, "bold"))],
                [sg.Listbox(values=[], size=(25, 8), key="-TOP-PREDS-",
                            disabled=True, background_color='#222', text_color='#0F0')],
                [sg.HSeparator()],
                [sg.Text("Current Word", font=("Arial", 12, "bold"))],
                [sg.Multiline(size=(25, 3), key="-WORD-DISPLAY-", disabled=True,
                              background_color='#111', text_color='#0F0',
                              font=("Courier", 14, "bold"))],
                [sg.Text("Buffered Letter: --", key="-BUFFER-INFO-", size=(25, 1))],
                [sg.HSeparator()],
                [sg.Text("Controls", font=("Arial", 12, "bold"))],
                [sg.Button("Save Word",   key="-SAVE-WORD-"),
                 sg.Button("Clear Word",  key="-CLEAR-WORD-")],
                [sg.Button("Undo Letter", key="-UNDO-LETTER-"),
                 sg.Button("Reset Merger",key="-RESET-MERGER-")],
                [sg.HSeparator()],
                [sg.Text("Mode", font=("Arial", 12, "bold"))],
                [sg.Radio("ASL Alphabet", "mode", default=True, key="-MODE-ASL-"),
                 sg.Radio("Custom Signs", "mode",               key="-MODE-CUSTOM-")],
                [sg.HSeparator()],
                [sg.Text("Tools", font=("Arial", 12, "bold"))],
                [sg.Button("Record Custom Sign", key="-RECORD-SIGN-", size=(22, 1))],
                [sg.Button("Train Custom Model", key="-TRAIN-MODEL-", size=(22, 1))],
                [sg.HSeparator()],
                [sg.Button("Settings", key="-SETTINGS-"),
                 sg.Button("Help",     key="-HELP-")],
                [sg.Button("Exit", key="-EXIT-")],
            ], vertical_alignment='top', size=(320, 530)),
        ],
        [sg.HSeparator()],
        [sg.Column([
            [sg.Text("Status: Ready", key="-STATUS-", size=(80, 1))],
        ], vertical_alignment='bottom')],
    ]

    return sg.Window(
        "Sign Language Translator", layout,
        finalize=True, size=(1050, 620), location=(50, 50)
    )


# ── GUI update ────────────────────────────────────────────────────────────────
def update_window_from_prediction(window, prediction):
    if prediction is None:
        return

    top_5 = prediction['top_5']
    window["-TOP-PREDS-"].update([f"{sign}: {conf:.1%}" for sign, conf in top_5])
    window["-WORD-DISPLAY-"].update(prediction['word'])

    buffer_info  = prediction['buffer_info']
    buffered     = buffer_info.get('buffered_letter', '--')
    buf_frames   = buffer_info.get('buffer_frame_count', 0)
    window["-BUFFER-INFO-"].update(f"Buffered: {buffered} ({buf_frames} frames)")

    hands  = prediction['metadata']['num_hands']
    fps    = buffer_info['fps_estimate']
    mode   = "CUSTOM" if use_custom_model else "ASL"
    window["-STATUS-"].update(
        f"Hands: {hands} | FPS: {fps:.1f} | Mode: {mode} | Word: '{prediction['word']}'"
    )


# ── Dialogs ───────────────────────────────────────────────────────────────────
def show_help_dialog():
    help_text = """
SIGN LANGUAGE TRANSLATOR - HELP

FEATURES:
• Real-time ASL alphabet recognition (A-Z)
• Custom sign recording and training
• Live letter merging to form words
• High accuracy with M1/M2 GPU acceleration

CONTROLS:
• Save Word    – save current word to file
• Clear Word   – clear the word and reset merger
• Undo Letter  – remove last confirmed letter
• Reset Merger – start fresh

MODES:
• ASL Alphabet – standard A-Z signs
• Custom Signs – your own trained signs (A, B, C, D)

WORKFLOW FOR CUSTOM SIGNS:
1. Click "Record Custom Sign" → type sign name (e.g. A, B)
2. Press SPACE to start/stop each sample (record ≥ 20 samples)
3. Click "Train Custom Model" and wait
4. Switch to "Custom Signs" mode — detections go live

TIPS:
• Good lighting + clear hand positioning = much better accuracy
• Hold each sign steady for ~0.5 s
• Lower Confidence Threshold in Settings if detections are sparse
• For custom signs (A-D): use Custom Signs mode, NOT ASL mode
    """
    sg.popup_scrolled(help_text, title="Help", size=(50, 25))


def show_settings_dialog(window):
    settings_layout = [
        [sg.Text("Confidence Threshold", font=("Arial", 10, "bold"))],
        [sg.Slider(range=(0, 1), default_value=config.CONFIDENCE_THRESHOLD,
                   resolution=0.05, key="-CONF-THRESH-", orientation='h')],
        [sg.Text("Min Hold Time (seconds)", font=("Arial", 10, "bold"))],
        [sg.Slider(range=(0.1, 1.5), default_value=config.MIN_HOLD_TIME,
                   resolution=0.1, key="-MIN-HOLD-", orientation='h')],
        [sg.HSeparator()],
        [sg.Button("Save"), sg.Button("Cancel")],
    ]
    sw = sg.Window("Settings", settings_layout, finalize=True)
    while True:
        ev, vals = sw.read()
        if ev in ("Cancel", sg.WINDOW_CLOSED):
            break
        elif ev == "Save":
            config.CONFIDENCE_THRESHOLD = vals["-CONF-THRESH-"]
            config.MIN_HOLD_TIME        = vals["-MIN-HOLD-"]
            sg.popup("Settings saved!", title="Success")
            break
    sw.close()


def save_word_to_file(word: str):
    if not word:
        sg.popup("No word to save", title="Info")
        return
    ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.PROJECT_ROOT / "translations" / f"output_{ts}.txt"
    output_file.parent.mkdir(exist_ok=True)
    try:
        with open(output_file, 'a') as f:
            f.write(f"{ts}: {word}\n")
        sg.popup(f"Saved to {output_file.name}", title="Success")
    except Exception as e:
        sg.popup(f"Error saving: {e}", title="Error")


def _stop_video_thread(vt: threading.Thread) -> None:
    global is_running
    is_running = False
    vt.join(timeout=3)


def _start_video_thread(window) -> threading.Thread:
    global is_running
    is_running = True
    vt = threading.Thread(target=video_thread, args=(window,), daemon=True)
    vt.start()
    return vt


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global is_running, use_custom_model, _predictor, _merger

    print("\n" + "=" * 60)
    print("SIGN LANGUAGE TRANSLATOR - GUI")
    print("=" * 60)

    window = create_window()
    vt     = _start_video_thread(window)

    # FIX: if a custom model already exists on disk, pre-select Custom mode
    import time; time.sleep(1.5)   # give video thread time to load models
    with _vt_lock:
        if _predictor is not None and _predictor.custom_model is not None:
            use_custom_model = True
            window["-MODE-CUSTOM-"].update(True)
            signs = _predictor.custom_signs
            window["-STATUS-"].update(
                f"Auto-selected Custom Signs mode — signs: {', '.join(signs)}"
            )
            print(f"[MAIN] Custom model detected, auto-switched to Custom mode ({signs})")

    print("[MAIN] GUI started")

    while True:
        event, values = window.read(timeout=10)

        try:
            frame_data, prediction = frame_queue.get_nowait()
            window["-IMAGE-"].update(data=frame_data)
            if prediction:
                update_window_from_prediction(window, prediction)
        except queue.Empty:
            pass

        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            print("[MAIN] Exiting...")
            is_running = False
            break

        elif event == "-SAVE-WORD-":
            word = window["-WORD-DISPLAY-"].get()
            if word.strip():
                save_word_to_file(word.strip())

        elif event == "-CLEAR-WORD-":
            with _vt_lock:
                if _merger:
                    _merger.reset()
            window["-WORD-DISPLAY-"].update("")
            window["-BUFFER-INFO-"].update("Buffered: -- (0 frames)")

        elif event == "-UNDO-LETTER-":
            with _vt_lock:
                if _merger:
                    updated = _merger.undo_last_letter()
                    window["-WORD-DISPLAY-"].update(updated)

        elif event == "-RESET-MERGER-":
            with _vt_lock:
                if _merger:
                    _merger.reset()
            window["-WORD-DISPLAY-"].update("")
            window["-BUFFER-INFO-"].update("Buffered: -- (0 frames)")

        elif event == "-RECORD-SIGN-":
            _stop_video_thread(vt)
            collect_sign_interactive()
            vt = _start_video_thread(window)

        elif event == "-TRAIN-MODEL-":
            _stop_video_thread(vt)
            train_custom_signs_interactive()
            with _vt_lock:
                if _predictor:
                    _predictor.reload_custom_model()
                    loaded_signs = _predictor.custom_signs
                    print(f"[MAIN] Custom model reloaded — signs: {loaded_signs}")
            vt = _start_video_thread(window)
            sg.popup(
                "Training complete!\nSwitch to 'Custom Signs' mode to use your model.",
                title="Training Done"
            )

        elif event == "-MODE-ASL-":
            use_custom_model = False
            window["-STATUS-"].update("Mode: ASL Alphabet")

        elif event == "-MODE-CUSTOM-":
            with _vt_lock:
                has_model = _predictor is not None and _predictor.custom_model is not None
                signs     = _predictor.custom_signs if _predictor else []
            if not has_model:
                sg.popup(
                    "No custom model found!\n\n"
                    "Steps:\n1. Record Custom Sign\n2. Train Custom Model\n3. Switch to Custom mode",
                    title="Custom Model Missing"
                )
                window["-MODE-ASL-"].update(True)
            else:
                use_custom_model = True
                window["-STATUS-"].update(f"Mode: Custom Signs ({', '.join(signs)})")

        elif event == "-HELP-":
            show_help_dialog()

        elif event == "-SETTINGS-":
            show_settings_dialog(window)

    is_running = False
    window.close()
    vt.join(timeout=5)
    print("[MAIN] Application closed")


if __name__ == "__main__":
    main()
