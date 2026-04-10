# src/quicktrain.py
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

# ── 1. Load ──────────────────────────────────────────────────────────────────
CSV_PATH = Path.home() / "Downloads" / "asl_landmarks_final.csv"  # fixed filename
df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} samples, columns: {list(df.columns[:5])} ... label={df.columns[-1]}")

label_col = df.columns[-1]           # last column is the label
X = df.drop(columns=[label_col]).values.astype(np.float32)
y = df[label_col].values

print(f"Feature shape: {X.shape}   Classes: {sorted(set(y))}")

# ── 2. Train ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = SVC(kernel="rbf", probability=True, C=10, gamma="scale")
print("Training SVM — takes ~30–60s on this dataset...")
clf.fit(X_train_s, y_train)

print(f"\nTest accuracy: {clf.score(X_test_s, y_test):.3f}")
print(classification_report(y_test, clf.predict(X_test_s)))

# ── 3. Save ───────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "models" / "asl_svm_model.pkl"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump({
        "model":        clf,
        "scaler":       scaler,
        "class_names":  sorted(set(y)),
        "feature_size": X.shape[1],   # 63 — saved so predictor knows
    }, f)
print(f"\nSaved → {out_path}")