import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Sequence

import cv2
import numpy as np
import tensorflow as tf


def list_classmask_files(folder: str) -> List[str]:
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and '_classmask' in f]
    files.sort()
    return [os.path.join(folder, f) for f in files]


def classmask_frame_id(path: str) -> str:
    """Estrae l'id frame DICOM dal nome della classmask, es. ..._000034_classmask.png."""
    match = re.search(r'_(\d+)_classmask(?:\.[^.]+)?$', Path(path).name)
    if not match:
        raise ValueError(f"Impossibile estrarre id frame da classmask: {path}")
    return match.group(1)


def list_classmask_frame_ids(folder: str) -> List[str]:
    return [classmask_frame_id(path) for path in list_classmask_files(folder)]


def list_dicom_frame_ids_in_reader_order(dicom_dir: str) -> List[str]:
    """Restituisce gli stem dei file DICOM nell'ordine usato da SimpleITK."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise ValueError(f"Nessun DICOM trovato in {dicom_dir}")
    return [Path(path).stem for path in dicom_names]


def select_x_slices_by_classmask_frames(
    x: np.ndarray,
    classmask_folder: str,
    dicom_dir: Optional[str] = None,
    frame_id_overrides: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Seleziona da X solo le fette DICOM per cui esiste una classmask.

    Se dicom_dir e' fornita, gli indici vengono calcolati sull'ordine di lettura
    SimpleITK; altrimenti si assume che l'id frame coincida con l'indice in X.
    """
    mask_frame_ids = list_classmask_frame_ids(classmask_folder)
    frame_id_overrides = frame_id_overrides or {}
    frame_ids = [frame_id_overrides.get(frame_id, frame_id) for frame_id in mask_frame_ids]
    if dicom_dir is not None:
        dicom_frame_ids = list_dicom_frame_ids_in_reader_order(dicom_dir)
        frame_to_idx = {frame_id: idx for idx, frame_id in enumerate(dicom_frame_ids)}
        missing = [frame_id for frame_id in frame_ids if frame_id not in frame_to_idx]
        if missing:
            raise ValueError(
                f"Classmask senza DICOM corrispondente in {dicom_dir}: {missing}"
            )
        indices = [frame_to_idx[frame_id] for frame_id in frame_ids]
    else:
        indices = [int(frame_id) for frame_id in frame_ids]

    out_of_range = [idx for idx in indices if idx >= x.shape[0]]
    if out_of_range:
        raise ValueError(
            f"Indici classmask fuori range per X con {x.shape[0]} fette: {out_of_range}"
        )

    frame_mapping = [
        f"mask {mask_frame_id}->DICOM {dicom_frame_id}->X[{idx}]"
        for mask_frame_id, dicom_frame_id, idx in zip(mask_frame_ids, frame_ids, indices)
    ]
    print("Frame RM selezionati dalle classmask:", ", ".join(frame_mapping))
    return x[indices]


def read_classmask_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Impossible leggere classmask: {path}")
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def classmask_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """Converti una classmask in binario: 0 sfondo, 1 foreground."""
    if mask_rgb.ndim == 2:
        return (mask_rgb > 0).astype(np.uint8)

    background = np.array([0, 0, 0], dtype=np.uint8)
    return np.any(mask_rgb != background, axis=-1).astype(np.uint8)


def resize_label(label_img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(label_img, target_size, interpolation=cv2.INTER_NEAREST)


def classmask_dir_to_label_stack(folder: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Carica tutti i classmask dalla cartella in un array di shape (N, H, W)."""
    files = list_classmask_files(folder)
    if not files:
        raise ValueError(f"Nessun file classmask trovato in {folder}")

    masks = []
    for path in files:
        img = read_classmask_image(path)
        label = classmask_to_label(img)
        if target_size is not None:
            label = resize_label(label, target_size)
        masks.append(label)

    return np.stack(masks, axis=0)


def label_stack_to_onehot(label_stack: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Converte uno stack di etichette intere in one-hot di shape (N, H, W, C)."""
    if label_stack.ndim != 3:
        raise ValueError('label_stack deve essere (N, H, W)')
    if num_classes is None:
        num_classes = int(label_stack.max()) + 1
    return tf.keras.utils.to_categorical(label_stack, num_classes)


def build_y_dataset(
    classmask_folder: str,
    target_size: Tuple[int, int] = (256, 256),
    one_hot: bool = True,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Restituisce y_train o y_val a partire da classmask PNG multi-classe."""
    labels = classmask_dir_to_label_stack(classmask_folder, target_size=target_size)
    if one_hot:
        return label_stack_to_onehot(labels, num_classes=num_classes)
    return labels


def build_x_dataset_from_cases(
    case_specs: Sequence[Tuple[str, str]],
) -> np.ndarray:
    """Costruisce X_train a partire da casi DICOM + maschera 3D NIfTI.

    case_specs: sequenza di (dicom_dir, mask_nifti_path).
    Restituisce un tensore (N, 256, 256, 2) concatenando tutte le fette.
    """
    import importlib.util
    import pathlib

    module_path = pathlib.Path(__file__).resolve().parents[1] / 'segmentazione_2.5D.py'
    spec = importlib.util.spec_from_file_location('segmentazione_2p5D', str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Impossibile importare {module_path}')
    seg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seg_mod)
    process_single_view = seg_mod.process_single_view

    slices = []
    for dicom_dir, mask_nifti_path in case_specs:
        slices.append(process_single_view(dicom_dir, mask_nifti_path))

    if not slices:
        raise ValueError('Nessun caso fornito per build_x_dataset_from_cases')

    return np.concatenate(slices, axis=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert classmask PNGs to label tensors')
    parser.add_argument('folder', help='Cartella contenente i file *_classmask.png')
    parser.add_argument('--output', '-o', default='y_labels.npy', help='File .npy di output')
    parser.add_argument('--height', type=int, default=256, help='Altezza target')
    parser.add_argument('--width', type=int, default=256, help='Larghezza target')
    parser.add_argument('--one-hot', action='store_true', help='Salva in formato one-hot (N, H, W, C)')
    args = parser.parse_args()

    y = build_y_dataset(args.folder, target_size=(args.width, args.height), one_hot=args.one_hot)
    np.save(args.output, y)
    print(f'Salvato {args.output} con shape {y.shape}')
