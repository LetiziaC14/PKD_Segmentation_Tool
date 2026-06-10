import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from model import build_modern_autokidneycyst_model
from preprocess import (
    build_y_dataset,
    build_x_dataset_from_cases,
    list_classmask_frame_ids,
    select_x_slices_by_classmask_frames,
)
from volume_data import build_full_volume_for_inference, build_volume_tensors


# --- CONFIGURATION ---
USE_MULTIVIEW_VOLUME = True
INPUT_CHANNELS = 4 if USE_MULTIVIEW_VOLUME else 2
SAVE_VOLUME_NIFTI = True

WEIGHTS_PATH = "best_autokidneycyst_weights.weights.h5"
TRAIN_CLASSMASK_FOLDER = "A/20240916-RM ABDOMEN(FP)/10001-COR T2 HASTE/masks"
VAL_CLASSMASK_FOLDER = None
TARGET_SIZE = (256, 256)

X_TRAIN_PATH = "X_train.npy"
X_VAL_PATH = "X_val.npy"

TRAIN_CASE_SPECS = [
    ("Dicom_Torra/10001-COR T2 HASTE", "Kidney_mask/kidney_mask_3d.nii.gz"),
]
VAL_CASE_SPECS = None

# Usa questo dizionario solo se una classmask e' stata salvata con un id diverso
# dal DICOM a cui appartiene, es. {"000051": "000052"}.
CLASSMASK_FRAME_OVERRIDES = {}

# Coerente con lo split manuale previsto in train.py.
TRAIN_SLICE_RANGE = slice(0, 7)
VAL_SLICE_RANGE = slice(7, 10)
INFERENCE_SUBSET = "all"  # "val", "train" oppure "all"

BATCH_SIZE = 8
NUM_VISUAL_SLICES = 5
OUTPUT_DIR = "inference_results"

# Segmentazione binaria cisti vs sfondo, coerente con train.py.
TASK_NUM_CLASSES = 1
PREDICTION_THRESHOLD = 0.5


def get_num_output_channels_from_weights(weights_path: str) -> int:
    """Legge il numero di canali del layer di output dall'archivio pesi Keras."""
    output_layers = []

    def register_output_kernel(layer_name: str, kernel: h5py.Dataset) -> None:
        if len(kernel.shape) == 4 and kernel.shape[0] == 1 and kernel.shape[1] == 1:
            if layer_name == "conv2d":
                layer_idx = 0
            else:
                layer_idx = int(layer_name.split("conv2d_")[1])
            output_layers.append((layer_idx, int(kernel.shape[-1])))

    with h5py.File(weights_path, "r") as weights_file:
        if "layers" in weights_file and isinstance(weights_file["layers"], h5py.Group):
            for layer_name in weights_file["layers"].keys():
                layer_group = weights_file["layers"][layer_name]
                if "vars" not in layer_group or "0" not in layer_group["vars"]:
                    continue
                register_output_kernel(layer_name, layer_group["vars"]["0"])
        else:
            for key in weights_file.keys():
                if not key.startswith("layers/conv2d_") or not key.endswith("/vars/0"):
                    continue
                layer_name = key.split("/")[1]
                register_output_kernel(layer_name, weights_file[key])

    if not output_layers:
        raise ValueError(f"Impossibile determinare i canali di output da {weights_path}")
    return max(output_layers, key=lambda item: item[0])[1]


def get_num_input_channels_from_weights(weights_path: str) -> int:
    with h5py.File(weights_path, "r") as weights_file:
        if "layers" in weights_file and "conv2d" in weights_file["layers"]:
            kernel = weights_file["layers"]["conv2d"]["vars"]["0"]
            return int(kernel.shape[2])
        for key in weights_file.keys():
            if key.endswith("conv2d/vars/0") or key == "layers/conv2d/vars/0":
                return int(weights_file[key].shape[2])
    raise ValueError(f"Impossibile determinare i canali di input da {weights_path}")


def resolve_model_num_classes(weights_path: str) -> int:
    """Verifica che i pesi salvati siano compatibili con il task binario."""
    weights_channels = get_num_output_channels_from_weights(weights_path)
    if weights_channels != TASK_NUM_CLASSES:
        raise ValueError(
            f"I pesi in '{weights_path}' hanno {weights_channels} canali di output, "
            f"ma il task attuale e' binario ({TASK_NUM_CLASSES} canale, sigmoid + dice). "
            "Probabilmente il modello e' stato addestrato con una versione precedente "
            "che trattava ogni cisti come classe separata. "
            "Riaddestra con train.py per generare pesi compatibili."
        )
    return TASK_NUM_CLASSES


def load_best_model(
    weights_path="best_autokidneycyst_weights.weights.h5",
    num_classes=1,
    input_channels=INPUT_CHANNELS,
):
    """Carica il modello e i pesi migliori salvati durante il training."""
    model = build_modern_autokidneycyst_model(
        img_rows=256,
        img_cols=256,
        num_classes=num_classes,
        input_channels=input_channels,
    )
    model.load_weights(weights_path)
    return model


def load_x_data(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File X non trovato: {path}")
    return np.load(path)


def load_full_dataset():
    """Carica il dataset completo da cui ricavare train/validation."""
    y = build_y_dataset(TRAIN_CLASSMASK_FOLDER, target_size=TARGET_SIZE, one_hot=False)
    frame_ids = list_classmask_frame_ids(TRAIN_CLASSMASK_FOLDER)

    if os.path.exists(X_TRAIN_PATH):
        x = load_x_data(X_TRAIN_PATH)
    elif TRAIN_CASE_SPECS:
        x = build_x_dataset_from_cases(TRAIN_CASE_SPECS)
    else:
        raise FileNotFoundError(
            "Imposta X_TRAIN_PATH o TRAIN_CASE_SPECS per generare X dai casi DICOM."
        )

    train_dicom_dir = TRAIN_CASE_SPECS[0][0] if len(TRAIN_CASE_SPECS) == 1 else None
    x = select_x_slices_by_classmask_frames(
        x,
        TRAIN_CLASSMASK_FOLDER,
        dicom_dir=train_dicom_dir,
        frame_id_overrides=CLASSMASK_FRAME_OVERRIDES,
    )

    min_slices = min(x.shape[0], y.shape[0])
    if x.shape[0] != y.shape[0]:
        print(f"Mismatch X/y: X ha {x.shape[0]} fette, y ha {y.shape[0]} fette.")
        print(f"Tronco a {min_slices} fette comuni.")
        x = x[:min_slices]
        y = y[:min_slices]

    dicom_frame_ids = [
        CLASSMASK_FRAME_OVERRIDES.get(frame_id, frame_id)
        for frame_id in frame_ids
    ]
    return x, y, dicom_frame_ids


def select_inference_subset(x, y, frame_ids, subset=INFERENCE_SUBSET):
    """Seleziona il subset su cui fare inferenza."""
    if subset == "val":
        x_subset = x[VAL_SLICE_RANGE]
        y_subset = y[VAL_SLICE_RANGE]
        selected_frame_ids = frame_ids[VAL_SLICE_RANGE]
        subset_description = f"validation manuale, fette {VAL_SLICE_RANGE.start}:{VAL_SLICE_RANGE.stop}"
    elif subset == "train":
        x_subset = x[TRAIN_SLICE_RANGE]
        y_subset = y[TRAIN_SLICE_RANGE]
        selected_frame_ids = frame_ids[TRAIN_SLICE_RANGE]
        subset_description = f"training manuale, fette {TRAIN_SLICE_RANGE.start}:{TRAIN_SLICE_RANGE.stop}"
    elif subset == "all":
        x_subset = x
        y_subset = y
        selected_frame_ids = frame_ids
        subset_description = "dataset completo"
    else:
        raise ValueError("INFERENCE_SUBSET deve essere 'val', 'train' oppure 'all'")

    if x_subset.shape[0] == 0:
        raise ValueError(
            f"Subset '{subset}' vuoto. Controlla VAL_SLICE_RANGE e numero di fette disponibili."
        )

    print(f"Inferenza su subset: {subset_description}")
    print("Frame inferenza:", ", ".join(selected_frame_ids))
    print(f"X_subset shape: {x_subset.shape}")
    print(f"y_subset shape: {y_subset.shape}")
    return x_subset, y_subset


def prepare_binary_y(y_labels: np.ndarray) -> np.ndarray:
    """Converte le etichette in (N, H, W, 1) float32, come in train.py."""
    if y_labels.ndim == 4 and y_labels.shape[-1] == 1:
        return y_labels.astype(np.float32)
    if y_labels.ndim != 3:
        raise ValueError("y deve avere shape (N, H, W) oppure (N, H, W, 1)")
    if int(np.max(y_labels)) > 1:
        raise ValueError(
            "Le classmask devono essere binarie (0=sfondo, 1=cisti). "
            "Rigenera y con preprocess.classmask_to_label."
        )
    return np.expand_dims(y_labels.astype(np.float32), axis=-1)


def predictions_to_labels(y_pred: np.ndarray, threshold: float = PREDICTION_THRESHOLD) -> np.ndarray:
    """Maschera binaria da output sigmoid (N, H, W, 1)."""
    if y_pred.shape[-1] != 1:
        raise ValueError(
            f"Output atteso con 1 canale, ricevuto shape {y_pred.shape}. "
            "Usa pesi addestrati con train.py (segmentazione binaria)."
        )
    return (y_pred[..., 0] > threshold).astype(np.uint8)


def predict_and_evaluate(model, x_eval, y_eval_one_hot):
    """Esegue predizioni e calcola metriche rispetto a y_eval_one_hot."""
    y_pred = model.predict(x_eval, batch_size=BATCH_SIZE)
    y_pred_labels = predictions_to_labels(y_pred)

    eval_result = model.evaluate(x_eval, y_eval_one_hot, batch_size=BATCH_SIZE, verbose=0)
    metric_names = model.metrics_names

    print("\n=== INFERENCE RESULTS ===")
    if isinstance(eval_result, list):
        for name, value in zip(metric_names, eval_result):
            print(f"{name}: {value:.4f}")
    else:
        print(f"{metric_names[0]}: {eval_result:.4f}")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions labels shape: {y_pred_labels.shape}")
    fg_fraction = float(np.mean(y_pred_labels > 0))
    print(f"Pixel foreground predetti: {fg_fraction * 100:.4f}%")
    if fg_fraction < 0.001:
        print(
            "ATTENZIONE: maschera predetta quasi vuota. "
            "Verifica che i pesi siano stati addestrati con train.py."
        )

    return y_pred, y_pred_labels


def make_mask_error_map(gt_label, pred_label):
    """Restituisce una mappa RGB: TP verde, FN blu, FP rosso."""
    gt_fg = gt_label > 0
    pred_fg = pred_label > 0

    comparison = np.zeros((*gt_label.shape, 3), dtype=np.float32)
    comparison[gt_fg & pred_fg] = (0.0, 0.75, 0.0)
    comparison[gt_fg & ~pred_fg] = (0.1, 0.35, 1.0)
    comparison[~gt_fg & pred_fg] = (1.0, 0.15, 0.1)
    return comparison


def visualize_predictions(x_eval, y_true_labels, y_pred_labels, num_slices=3):
    """Salva confronto tra input, maschera reale, maschera predetta ed errori."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for slice_idx in range(min(num_slices, x_eval.shape[0])):
        mr_image = x_eval[slice_idx, :, :, 0]
        gt_label = (y_true_labels[slice_idx] > 0).astype(np.uint8)
        pred_label = (y_pred_labels[slice_idx] > 0).astype(np.uint8)
        error_map = make_mask_error_map(gt_label, pred_label)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(mr_image, cmap="gray")
        axes[0].set_title(f"Input RM - slice {slice_idx}")
        axes[0].axis("off")

        im_gt = axes[1].imshow(gt_label, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Maschera reale")
        axes[1].axis("off")
        plt.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

        im_pred = axes[2].imshow(pred_label, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Maschera predetta")
        axes[2].axis("off")
        plt.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(mr_image, cmap="gray")
        axes[3].imshow(error_map, alpha=0.55)
        axes[3].set_title("Confronto: verde ok, blu mancata, rosso extra")
        axes[3].axis("off")

        output_path = os.path.join(OUTPUT_DIR, f"slice_{slice_idx:03d}_pred_vs_gt.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"Salvato: {output_path}")

    print(f"Visualizzazioni salvate in '{OUTPUT_DIR}/'")


def save_prediction_volume(pred_labels: np.ndarray, reference_image: sitk.Image, output_path: str):
    volume = sitk.GetImageFromArray(pred_labels.astype(np.uint8))
    volume.CopyInformation(reference_image)
    sitk.WriteImage(volume, output_path)
    print(f"Volume predetto salvato: {output_path}")


def load_multiview_eval_set():
    """Carica fette supervise (GT fuso AX/COR/SAG) per valutazione."""
    x_all, y_all, axial_master, z_indices = build_volume_tensors(supervised_only=True)
    n_val = max(1, int(round(0.2 * x_all.shape[0])))
    x_eval = x_all[:n_val]
    y_eval = y_all[:n_val]
    print(f"Eval multi-vista: {x_eval.shape[0]} fette, Z={z_indices[:n_val].tolist()}")
    return x_eval, y_eval, axial_master, z_indices[:n_val]


def main():
    num_classes = resolve_model_num_classes(WEIGHTS_PATH)
    weight_input_channels = get_num_input_channels_from_weights(WEIGHTS_PATH)
    if weight_input_channels != INPUT_CHANNELS:
        raise ValueError(
            f"I pesi richiedono {weight_input_channels} canali di input, "
            f"ma USE_MULTIVIEW_VOLUME imposta {INPUT_CHANNELS}. Riaddestra con train.py."
        )

    if USE_MULTIVIEW_VOLUME:
        if INFERENCE_SUBSET == "full_volume":
            x_eval, axial_master, _ = build_full_volume_for_inference()
            y_eval_labels = None
            print(f"Inferenza volume completo: {x_eval.shape}")
        else:
            x_eval, y_eval_labels, axial_master, _ = load_multiview_eval_set()
    else:
        if VAL_CLASSMASK_FOLDER or VAL_CASE_SPECS or os.path.exists(X_VAL_PATH):
            raise NotImplementedError(
                "Validation set separato non supportato in modalita' legacy."
            )
        x_all, y_all_labels, frame_ids = load_full_dataset()
        x_eval, y_eval_labels = select_inference_subset(x_all, y_all_labels, frame_ids)
        axial_master = None

    model = load_best_model(
        WEIGHTS_PATH,
        num_classes=num_classes,
        input_channels=INPUT_CHANNELS,
    )

    y_eval_one_hot = None
    if y_eval_labels is not None:
        y_eval_one_hot = prepare_binary_y(y_eval_labels)
        y_pred, y_pred_labels = predict_and_evaluate(model, x_eval, y_eval_one_hot)
    else:
        y_pred = model.predict(x_eval, batch_size=BATCH_SIZE)
        y_pred_labels = predictions_to_labels(y_pred)
        print(f"Predictions shape: {y_pred.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)
    np.save(os.path.join(OUTPUT_DIR, "y_pred_labels.npy"), y_pred_labels)

    if USE_MULTIVIEW_VOLUME and SAVE_VOLUME_NIFTI and INFERENCE_SUBSET == "full_volume":
        save_prediction_volume(
            y_pred_labels,
            axial_master,
            os.path.join(OUTPUT_DIR, "cyst_pred_volume.nii.gz"),
        )

    print(f"\nPrevisioni salvate in: {OUTPUT_DIR}/")

    if y_eval_labels is not None:
        print("\n=== VISUALIZATION ===")
        visualize_predictions(
            x_eval,
            y_eval_labels,
            y_pred_labels,
            num_slices=NUM_VISUAL_SLICES,
        )


if __name__ == "__main__":
    main()
