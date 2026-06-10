import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from model import build_modern_autokidneycyst_model
from preprocess import (
    build_x_dataset_from_cases,
    build_y_dataset,
    list_classmask_frame_ids,
    select_x_slices_by_classmask_frames,
)
from volume_data import build_volume_tensors

# --- DATA CONFIGURATION ---
USE_MULTIVIEW_VOLUME = True
INPUT_CHANNELS = 4 if USE_MULTIVIEW_VOLUME else 2

# Legacy coronal-only (USE_MULTIVIEW_VOLUME = False)
TRAIN_CLASSMASK_FOLDER = "A/20240916-RM ABDOMEN(FP)/10001-COR T2 HASTE/masks"
VAL_CLASSMASK_FOLDER = None
TARGET_SIZE = (256, 256)
X_TRAIN_PATH = "X_train.npy"
X_VAL_PATH = "X_val.npy"
TRAIN_CASE_SPECS = [
    ("Dicom_Torra/10001-COR T2 HASTE", "Kidney_mask/kidney_mask_3d.nii.gz"),
]
VAL_CASE_SPECS = None
CLASSMASK_FRAME_OVERRIDES = {}
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def load_x_data(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File X non trovato: {path}")
    return np.load(path)


def split_train_val(x, y, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"Servono almeno 2 fette per train/val, disponibili {n}.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        train_idx = val_idx[1:]
        val_idx = val_idx[:1]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


if USE_MULTIVIEW_VOLUME:
    print("=== MULTI-VIEW VOLUME PIPELINE (AX + SAG + COR) ===")
    X_all, y_all, axial_master, z_indices = build_volume_tensors(supervised_only=True)
    X_train, y_train, X_val, y_val = split_train_val(X_all, y_all)
    print(f"Fette assiali supervise (Z): {z_indices.tolist()}")
else:
    y_train = build_y_dataset(TRAIN_CLASSMASK_FOLDER, target_size=TARGET_SIZE, one_hot=False)
    y_val = None
    if VAL_CLASSMASK_FOLDER:
        y_val = build_y_dataset(VAL_CLASSMASK_FOLDER, target_size=TARGET_SIZE, one_hot=False)

    if os.path.exists(X_TRAIN_PATH):
        X_train = load_x_data(X_TRAIN_PATH)
    elif TRAIN_CASE_SPECS:
        X_train = build_x_dataset_from_cases(TRAIN_CASE_SPECS)
    else:
        raise FileNotFoundError(
            "Imposta X_TRAIN_PATH o TRAIN_CASE_SPECS per generare X_train dai casi DICOM."
        )

    X_val = None
    if os.path.exists(X_VAL_PATH):
        X_val = load_x_data(X_VAL_PATH)
    elif VAL_CASE_SPECS:
        X_val = build_x_dataset_from_cases(VAL_CASE_SPECS)

    if VAL_CLASSMASK_FOLDER is None and VAL_CASE_SPECS is None and X_val is None:
        train_dicom_dir = TRAIN_CASE_SPECS[0][0] if len(TRAIN_CASE_SPECS) == 1 else None
        X_train = select_x_slices_by_classmask_frames(
            X_train,
            TRAIN_CLASSMASK_FOLDER,
            dicom_dir=train_dicom_dir,
            frame_id_overrides=CLASSMASK_FRAME_OVERRIDES,
        )

    min_slices = min(X_train.shape[0], y_train.shape[0])
    if X_train.shape[0] != y_train.shape[0]:
        print(f"Mismatch: X_train {X_train.shape[0]} fette, y_train {y_train.shape[0]} fette.")
        X_train = X_train[:min_slices]
        y_train = y_train[:min_slices]

    if X_val is None and y_val is None:
        train_slice = slice(0, 7)
        val_slice = slice(7, 10)
        classmask_frame_ids = list_classmask_frame_ids(TRAIN_CLASSMASK_FOLDER)
        if X_train.shape[0] < val_slice.stop or y_train.shape[0] < val_slice.stop:
            raise ValueError(
                f"Servono almeno {val_slice.stop} fette per lo split manuale; "
                f"disponibili X={X_train.shape[0]}, y={y_train.shape[0]}."
            )
        X_all, y_all = X_train, y_train
        X_train, X_val = X_all[train_slice], X_all[val_slice]
        y_train, y_val = y_all[train_slice], y_all[val_slice]
        print("Frame training:", ", ".join(classmask_frame_ids[train_slice]))
        print("Frame validation:", ", ".join(classmask_frame_ids[val_slice]))

print(f"X_train {X_train.shape}, X_val {X_val.shape}")
print(f"y_train {y_train.shape}, y_val {y_val.shape}")

if X_val is None or y_val is None:
    raise ValueError("X_val e y_val devono essere definiti.")

if y_train.ndim == 3:
    if int(max(np.max(y_train), np.max(y_val))) > 1:
        raise ValueError("Le classmask devono essere binarie (0/1).")
    num_classes = 1
    y_train = np.expand_dims(y_train.astype(np.float32), axis=-1)
    y_val = np.expand_dims(y_val.astype(np.float32), axis=-1)
elif y_train.ndim == 4 and y_train.shape[-1] == 1:
    num_classes = 1
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
else:
    raise ValueError("y_train deve avere shape (N, H, W) oppure (N, H, W, 1)")

if X_train.shape[-1] != INPUT_CHANNELS:
    raise ValueError(
        f"Canali input attesi: {INPUT_CHANNELS}, ricevuto X_train {X_train.shape}."
    )

model = build_modern_autokidneycyst_model(
    img_rows=256,
    img_cols=256,
    num_classes=num_classes,
    input_channels=INPUT_CHANNELS,
)

checkpoint_cb = ModelCheckpoint(
    filepath="best_autokidneycyst_weights.weights.h5",
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=200,
    callbacks=[checkpoint_cb],
)
