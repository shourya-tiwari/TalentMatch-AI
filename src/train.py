"""
TalentMatch AI — ANN Training Pipeline
=======================================
Architecture:
  Input(13) → Dense(64, ReLU) → Dropout(0.3)
            → Dense(32, ReLU) → Dropout(0.2)
            → Dense(16, ReLU)
            → Dense(1, Sigmoid)

Training strategy:
  - Binary cross-entropy loss
  - Adam optimizer
  - EarlyStopping on val_loss (patience=10)
  - ReduceLROnPlateau on val_loss (patience=5)
  - ModelCheckpoint saves best weights only
  - Stratified train/val/test split (70/15/15)

Artifacts saved to models/:
  - talentmatch_ann_v1.keras     (best model weights)
  - training_history_v1.json     (loss/accuracy per epoch)
  - training_config_v1.json      (hyperparameters, split sizes, feature list)
  - label_encoder_v1.joblib      (class weights if imbalanced)
"""

import os
import warnings

# 1. Suppress TensorFlow native INFO and WARNING logs (3 = Errors only)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Suppress oneDNN custom operations floating-point warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Suppress standard Python/absl warnings
warnings.filterwarnings('ignore')


import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, callbacks # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
CONFIG = {
    "model_version":    "v1",
    "seed":             SEED,
    "test_size":        0.15,
    "val_size":         0.15,
    "batch_size":       64,
    "max_epochs":       150,
    "learning_rate":    0.001,
    "dropout_1":        0.3,
    "dropout_2":        0.2,
    "dense_units":      [64, 32, 16],
    "activation":       "relu",
    "loss":             "binary_crossentropy",
    "optimizer":        "adam",
    "early_stop_patience":  10,
    "reduce_lr_patience":   5,
    "reduce_lr_factor":     0.5,
    "reduce_lr_min":        1e-6,
    "fit_label_threshold":  50,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data():
    """
    Load model_training_dataset.csv.
    Returns X (features), y (labels), feature column names.
    """
    path = PROC_DIR / "model_training_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"model_training_dataset.csv not found at {path}\n"
            "Run src/run_feature_pipeline.py first."
        )

    df = pd.read_csv(path)

    # Drop non-feature identifier columns
    drop_cols = ["application_id", "candidate_id", "job_id", "fit_score", "fit_label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["fit_label"].values.astype(np.float32)

    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class balance — Positive: {y.sum():.0f} ({y.mean()*100:.1f}%) "
          f"| Negative: {(1-y).sum():.0f} ({(1-y.mean())*100:.1f}%)")

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Data Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray):
    """
    Stratified split into train / val / test sets.
    Ratios: 70% train, 15% val, 15% test.
    Stratification preserves class balance across all splits.
    """
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=SEED,
        stratify=y
    )

    # Split remaining into train and val
    val_ratio_adjusted = CONFIG["val_size"] / (1.0 - CONFIG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        random_state=SEED,
        stratify=y_train_val
    )

    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights to handle label imbalance.
    Even mild imbalance benefits from this — it costs nothing.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Class weights: {class_weight_dict}")
    return class_weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> keras.Model:
    """
    Build TalentMatch ANN.

    Architecture rationale:
      - 3 hidden layers with decreasing width: forces hierarchical compression
      - ReLU avoids vanishing gradient on hidden layers
      - Dropout after first two layers: regularisation against overfitting
      - Sigmoid output: probability of good fit (0–1)
      - He normal initialisation: correct for ReLU activations
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,), name="input"),

        layers.Dense(
            CONFIG["dense_units"][0],
            activation=CONFIG["activation"],
            kernel_initializer="he_normal",
            name="hidden_1"
        ),
        layers.Dropout(CONFIG["dropout_1"], name="dropout_1"),

        layers.Dense(
            CONFIG["dense_units"][1],
            activation=CONFIG["activation"],
            kernel_initializer="he_normal",
            name="hidden_2"
        ),
        layers.Dropout(CONFIG["dropout_2"], name="dropout_2"),

        layers.Dense(
            CONFIG["dense_units"][2],
            activation=CONFIG["activation"],
            kernel_initializer="he_normal",
            name="hidden_3"
        ),

        layers.Dense(1, activation="sigmoid", name="output")
    ], name="TalentMatch_ANN_v1")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss=CONFIG["loss"],
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks() -> list:
    """
    EarlyStopping    — stop when val_loss stops improving (patience=10)
    ReduceLROnPlateau — halve LR when val_loss stagnates (patience=5)
    ModelCheckpoint  — save only the best model (by val_loss)
    """
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["early_stop_patience"],
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=CONFIG["reduce_lr_factor"],
        patience=CONFIG["reduce_lr_patience"],
        min_lr=CONFIG["reduce_lr_min"],
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "talentmatch_ann_v1.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    return [early_stop, reduce_lr, checkpoint]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X_train, X_val,
    y_train, y_val,
    model: keras.Model,
    class_weights: dict
) -> keras.callbacks.History:
    """Run the training loop."""

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["max_epochs"],
        batch_size=CONFIG["batch_size"],
        class_weight=class_weights,
        callbacks=build_callbacks(),
        verbose=1
    )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, history, feature_cols, split_sizes):
    """Save model, training history, and config to models/."""

    # Training history
    history_path = MODEL_DIR / "training_history_v1.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"  Saved history  → {history_path}")

    # Training config + metadata
    config_out = CONFIG.copy()
    config_out["feature_columns"]  = feature_cols
    config_out["n_features"]       = len(feature_cols)
    config_out["train_samples"]    = split_sizes["train"]
    config_out["val_samples"]      = split_sizes["val"]
    config_out["test_samples"]     = split_sizes["test"]
    config_out["epochs_trained"]   = len(history.history["loss"])
    config_out["trained_at"]       = datetime.now().isoformat()

    config_path = MODEL_DIR / "training_config_v1.json"
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"  Saved config   → {config_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("TalentMatch AI — ANN Training Pipeline")
    print("=" * 42)

    print("\n[1/5] Loading training data...")
    X, y, feature_cols = load_training_data()

    print("\n[2/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\n[3/5] Computing class weights...")
    class_weights = compute_weights(y_train)

    print("\n[4/5] Building model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    print("\n[5/5] Training...")
    history = train(X_train, X_val, y_train, y_val, model, class_weights)

    print("\n── Saving artifacts...")
    save_artifacts(
        model, history, feature_cols,
        split_sizes={
            "train": len(X_train),
            "val":   len(X_val),
            "test":  len(X_test)
        }
    )

    # Save test set for evaluate.py
    np.save(MODEL_DIR / "X_test_v1.npy", X_test)
    np.save(MODEL_DIR / "y_test_v1.npy", y_test)
    print(f"  Saved test set → models/X_test_v1.npy, y_test_v1.npy")

    # Final val metrics
    print("\n── Final Validation Metrics ─────────────────────")
    val_results = model.evaluate(X_val, y_val, verbose=0)
    metric_names = ["loss", "accuracy", "auc", "precision", "recall"]
    for name, val in zip(metric_names, val_results):
        print(f"   {name:<12}: {val:.4f}")
    print("─" * 50)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()