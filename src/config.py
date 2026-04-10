"""
Configuration file for Sign Language Translator
Optimized for MacOS M1/M2 with 8GB RAM

FIX: CONFIDENCE_THRESHOLD lowered to 0.35 for SVM custom model
     (SVM with 4 classes outputs confident probabilities, but 0.2 was
      accidentally too close to the random baseline of 0.25 for 4 classes,
      causing flickering. 0.35 filters noise without blocking real detections.)
     MIN_HOLD_TIME raised to 0.4s so that brief ambiguous frames don't commit.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ASL_DATA_DIR = DATA_DIR / "asl_alphabet"
CUSTOM_DATA_DIR = DATA_DIR / "custom_signs"
LANDMARKS_DIR = ASL_DATA_DIR / "landmarks"

# Model directories
MODEL_DIR = PROJECT_ROOT / "models"
ASL_MODEL_PATH = MODEL_DIR / "asl_svm_model.pkl"
CUSTOM_MODEL_PATH = MODEL_DIR / "custom_signs_model.h5"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"

# Docs directories
DOCS_DIR = PROJECT_ROOT / "docs"

# Create directories if they don't exist
for directory in [DATA_DIR, ASL_DATA_DIR, CUSTOM_DATA_DIR, LANDMARKS_DIR, MODEL_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MACOS M1/M2 OPTIMIZATION SETTINGS
# ============================================================================

ENABLE_METAL_GPU = True
TENSORFLOW_VERSION = "2.18"
BATCH_SIZE = 4
PREFETCH_SIZE = 2
ENABLE_MIXED_PRECISION = True

# ============================================================================
# LANDMARK EXTRACTION SETTINGS
# ============================================================================

HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7
POSE_DETECTION_CONFIDENCE = 0.7

HAND_LANDMARKS_COUNT = 21
POSE_LANDMARKS_COUNT = 33

# 21 landmarks * 3 coords (x,y,z) * 2 hands = 126
FEATURE_VECTOR_SIZE = 126

# ============================================================================
# LSTM MODEL ARCHITECTURE
# ============================================================================

MAX_SEQUENCE_LENGTH = 30
MIN_SEQUENCE_LENGTH = 5
FRAME_SKIP = 1

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 128
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001

NUM_ASL_SIGNS = 26
NUM_CUSTOM_SIGNS = 0

# ============================================================================
# LETTER MERGING SETTINGS
# ============================================================================

# FIX: 0.35 works much better for a 4-class SVM than 0.2.
# The SVM's softmax-style probabilities for correct predictions typically
# land between 0.6–0.99. At 0.2 you were confirming noise.
# At 0.35 you filter out uncertain frames while still catching real signs.
# (If you add more signs and confidence drops, lower this back toward 0.25.)
CONFIDENCE_THRESHOLD = 0.35

# FIX: 0.4 s gives the merger enough frames to accumulate a consistent mean
# before committing — reduces false positives without noticeable lag.
MIN_HOLD_TIME = 0.4

MAX_HOLD_TIME = 2.0

EXPECTED_FPS = 30
MIN_HOLD_FRAMES = int(MIN_HOLD_TIME * EXPECTED_FPS)   # ~12 frames
MAX_HOLD_FRAMES = int(MAX_HOLD_TIME * EXPECTED_FPS)   # ~60 frames

# ============================================================================
# VIDEO CAPTURE SETTINGS
# ============================================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
RESIZE_FRAME = (FRAME_WIDTH, FRAME_HEIGHT)

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

AUGMENT_DATA = True
AUGMENTATION_FACTOR = 1.5
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
VALIDATION_FREQUENCY = 2

# ============================================================================
# CUSTOM SIGNS RECORDING
# ============================================================================

SAMPLES_PER_SIGN = 20
RECORDING_DURATION = 2
MIN_FRAMES_PER_SAMPLE = 15

# ============================================================================
# GUI SETTINGS
# ============================================================================

GUI_THEME = 'DarkBlue3'
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
CONFIDENCE_DECIMALS = 2

# ============================================================================
# ASL ALPHABET MAPPING
# ============================================================================

ASL_SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

SIGN_TO_INDEX = {sign: i for i, sign in enumerate(ASL_SIGNS)}
INDEX_TO_SIGN = {i: sign for i, sign in enumerate(ASL_SIGNS)}

# ============================================================================
# DEBUG & LOGGING
# ============================================================================

DEBUG = False
SAVE_LOGS = True
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DEVICE DETECTION (MacOS M1/M2 specific)
# ============================================================================

def get_device_info():
    print("[CONFIG] Loading TensorFlow... (first run may take 30-60s)")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        device = "GPU (1 devices)" if gpus else "CPU"
    except ImportError:
        device = "TensorFlow not available"
    print(f"[CONFIG] TensorFlow loaded ✓  →  device: {device}")
    return device

DEVICE = get_device_info()

print(f"[CONFIG] Running on:   {DEVICE}")
print(f"[CONFIG] Project root: {PROJECT_ROOT}")
print(f"[CONFIG] Model dir:    {MODEL_DIR}")
print(f"[CONFIG] Data dir:     {DATA_DIR}")
print(f"[CONFIG] Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"[CONFIG] Min hold time: {MIN_HOLD_TIME}s ({MIN_HOLD_FRAMES} frames)")
print(f"[CONFIG] Config ready ✓")
