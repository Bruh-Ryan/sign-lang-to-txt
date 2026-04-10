
"""
LSTM Predictor Module
Supports both Keras LSTM (ASL) and SVM pickle (custom signs).


FIX APPLIED: _predict_custom now uses the RAW landmarks_sequence (before zero-padding)
to compute the representative frame, so the SVM gets clean signal instead of
a zero-dominated median.
"""


import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Dict
import config
from pathlib import Path
import json
import pickle
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)



class LSTMPredictor:


    def __init__(self, model_path: Optional[str] = None,
                 custom_model_path: Optional[str] = None):
        self.model         = None
        self.asl_model     = None
        self.custom_model  = None
        self.custom_scaler = None
        self.is_svm        = False
        self.model_config  = None
        self.sign_labels   = config.ASL_SIGNS.copy()
        self.custom_signs  = []
        self.is_using_custom = False


        # FIX: keep the raw (unpadded) landmark sequence so SVM gets clean input
        self._raw_sequence: List[np.ndarray] = []


        if model_path is None:
            model_path = str(config.ASL_MODEL_PATH)
        if custom_model_path is None:
            custom_model_path = str(config.CUSTOM_MODEL_PATH)


        self.load_model(model_path, custom_model_path)


    def load_model(self, model_path: str, custom_model_path: str):
        # ── ASL Keras model ──
        if Path(model_path).exists():
            try:
                print(f"[PREDICTOR] Loading ASL model from {model_path}...")
                self.asl_model = tf.keras.models.load_model(model_path)
                self.model     = self.asl_model
                print("[PREDICTOR] ASL model loaded")
            except Exception as e:
                print(f"[PREDICTOR] Error: {e} — using dummy model")
                self.asl_model = self._build_dummy_model()
                self.model     = self.asl_model
        else:
            print("[PREDICTOR] ASL model not found — using dummy model")
            self.asl_model = self._build_dummy_model()
            self.model     = self.asl_model


        # ── Custom model: .pkl (SVM) takes priority over .keras ──
        pkl_path   = str(custom_model_path).replace(".keras", ".pkl").replace(".h5", ".pkl")
        keras_path = str(custom_model_path).replace(".pkl", ".keras").replace(".h5", ".keras")


        if Path(pkl_path).exists():
            self._load_svm(pkl_path)
        elif Path(keras_path).exists():
            self._load_keras_custom(keras_path)


        # ── model_config.json ──
        if Path(config.MODEL_CONFIG_PATH).exists():
            try:
                with open(config.MODEL_CONFIG_PATH) as f:
                    self.model_config = json.load(f)
                if self.model_config.get("class_names"):
                    self.custom_signs = self.model_config["class_names"]
            except Exception as e:
                print(f"[PREDICTOR] Config load error: {e}")


    def _load_svm(self, pkl_path: str):
        try:
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            self.custom_model  = bundle["model"]
            self.custom_scaler = bundle["scaler"]
            self.custom_signs  = bundle.get("class_names", [])
            self.is_svm        = True
            print(f"[PREDICTOR] SVM model loaded — signs: {self.custom_signs}")
        except Exception as e:
            print(f"[PREDICTOR] SVM load error: {e}")


    def _load_keras_custom(self, keras_path: str):
        try:
            self.custom_model = tf.keras.models.load_model(keras_path)
            self.is_svm       = False
            self._load_custom_signs_metadata()
            print(f"[PREDICTOR] Keras custom model loaded")
        except Exception as e:
            print(f"[PREDICTOR] Keras custom load error: {e}")


    def _build_dummy_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(config.MAX_SEQUENCE_LENGTH, config.FEATURE_VECTOR_SIZE)),
            tf.keras.layers.LSTM(config.LSTM_UNITS_1, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(config.LSTM_UNITS_2, activation="relu"),
            tf.keras.layers.Dense(config.DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dropout(config.DROPOUT_RATE),
            tf.keras.layers.Dense(config.NUM_ASL_SIGNS, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model


    def predict(self, landmarks_sequence: List[np.ndarray]
                ) -> Tuple[str, float, Dict[str, float]]:
        """
        landmarks_sequence: list of raw (126,) arrays, NO zero-padding yet.
        We keep this raw list for SVM and only pad for the Keras path.
        """
        if not landmarks_sequence:
            return "NONE", 0.0, {}


        # Store raw (unpadded) sequence for SVM use
        self._raw_sequence = landmarks_sequence


        sequence = self._prepare_sequence(landmarks_sequence)   # (1, 30, 126)


        if self.is_using_custom and self.custom_model is not None:
            probabilities = self._predict_custom(sequence)
            labels = self.custom_signs
        else:
            probabilities = self.model.predict(sequence, verbose=0)[0]
            labels = self.sign_labels


        all_probs = {label: float(probabilities[i])
                     for i, label in enumerate(labels) if i < len(probabilities)}


        predicted_idx  = int(np.argmax(probabilities[:len(labels)]))
        predicted_sign = labels[predicted_idx] if predicted_idx < len(labels) else "UNKNOWN"
        confidence     = float(probabilities[predicted_idx])
        return predicted_sign, confidence, all_probs


    def _predict_custom(self, sequence: np.ndarray) -> np.ndarray:
        """
        FIX: For SVM, compute the representative feature vector from the RAW
        (unpadded) frames only — not the zero-padded tensor.
        Using mean instead of median gives slightly more stable results for
        static signs like A-D where the hand barely moves.
        """
        if self.is_svm:
            if len(self._raw_sequence) > 0:
                raw = np.array(self._raw_sequence, dtype=np.float32)  # (N_real, 126)
                flat = np.mean(raw, axis=0, keepdims=True)             # (1, 126)
            else:
                flat = np.mean(sequence[0], axis=0, keepdims=True)    # fallback


            scaled = self.custom_scaler.transform(flat)
            return self.custom_model.predict_proba(scaled)[0]


        # Keras custom model — use padded sequence as usual
        return self.custom_model.predict(sequence, verbose=0)[0]


    def predict_batch(self, landmarks_sequences):
        return [self.predict(s) for s in landmarks_sequences]


    def _prepare_sequence(self, landmarks_sequence: List[np.ndarray]) -> np.ndarray:
        sequence = np.array(landmarks_sequence, dtype=np.float32)
        if len(sequence) > config.MAX_SEQUENCE_LENGTH:
            sequence = sequence[-config.MAX_SEQUENCE_LENGTH:]
        if len(sequence) < config.MAX_SEQUENCE_LENGTH:
            padding  = np.zeros((config.MAX_SEQUENCE_LENGTH - len(sequence),
                                 config.FEATURE_VECTOR_SIZE), dtype=np.float32)
            sequence = np.vstack([padding, sequence])
        return np.expand_dims(sequence, axis=0)


    def get_top_predictions(self, all_probabilities: Dict[str, float],
                            top_k: int = 5) -> List[Tuple[str, float]]:
        return sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]


    def switch_to_custom_model(self):
        if self.custom_model is not None:
            self.is_using_custom = True
            self._load_custom_signs_metadata()
            kind = "SVM" if self.is_svm else "Keras"
            print(f"[PREDICTOR] Custom model active ({kind}) — {self.custom_signs}")
            return True
        print("[PREDICTOR] No custom model loaded")
        return False


    def switch_to_asl_model(self):
        self.model           = self.asl_model
        self.is_using_custom = False
        print("[PREDICTOR] ASL model active")
        return True


    def reload_custom_model(self):
        pkl_path   = str(config.CUSTOM_MODEL_PATH).replace(".keras", ".pkl").replace(".h5", ".pkl")
        keras_path = str(config.CUSTOM_MODEL_PATH).replace(".pkl", ".keras").replace(".h5", ".keras")
        if Path(pkl_path).exists():
            self._load_svm(pkl_path)
            return True
        elif Path(keras_path).exists():
            self._load_keras_custom(keras_path)
            return True
        print("[PREDICTOR] No custom model found to reload")
        return False


    def _load_custom_signs_metadata(self):
        try:
            with open(config.MODEL_CONFIG_PATH) as f:
                cfg = json.load(f)
            if cfg.get("class_names"):
                self.custom_signs = cfg["class_names"]
                return
        except Exception:
            pass
        meta = config.CUSTOM_DATA_DIR / "metadata.json"
        if meta.exists():
            try:
                with open(meta) as f:
                    self.custom_signs = json.load(f).get("signs", [])
            except Exception:
                pass


    def save_model_config(self, config_dict: Dict):
        with open(config.MODEL_CONFIG_PATH, "w") as f:
            json.dump(config_dict, f, indent=2)


    def get_model_info(self) -> Dict:
        return {
            "model_loaded":        self.model is not None,
            "custom_model_loaded": self.custom_model is not None,
            "custom_model_type":   "svm" if self.is_svm else "keras",
            "num_asl_signs":       len(self.sign_labels),
            "num_custom_signs":    len(self.custom_signs),
            "is_using_custom":     self.is_using_custom,
            "active_labels":       self.custom_signs if self.is_using_custom else self.sign_labels,
        }



if __name__ == "__main__":
    predictor = LSTMPredictor()
    seq = [np.random.randn(config.FEATURE_VECTOR_SIZE).astype(np.float32) for _ in range(15)]
    sign, conf, probs = predictor.predict(seq)
    print(f"Predicted: {sign} ({conf:.2f})")
    for s, p in predictor.get_top_predictions(probs):
        print(f"  {s}: {p:.4f}")