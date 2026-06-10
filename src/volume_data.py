"""Costruisce tensore multi-vista 3D e GT cisti fuso da annotazioni AX/COR/SAG."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from preprocess import classmask_to_label, list_classmask_files, read_classmask_image
from volume_config import (
    AXIAL_MRI_SERIES,
    CLASSMASK_SERIES_DIRS,
    CORONAL_MRI_SERIES,
    CYST_MASK_DIRS,
    KIDNEY_MASK_DILATE_RADIUS,
    KIDNEY_MASK_DIRS,
    SAGITTAL_MRI_SERIES,
    TARGET_SLICE_SIZE,
    TARGET_SPACING,
)


def _list_series_frame_paths(series_dir: Path) -> List[Path]:
    png_files = sorted(series_dir.glob("*.png"))
    png_files = [
        path for path in png_files
        if "_classmask" not in path.name and "_masks_overlay" not in path.name
    ]
    if not png_files:
        raise ValueError(f"Nessuna immagine serie in {series_dir}")
    return png_files


def load_png_series_as_volume(series_dir: Path, pixel_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Carica una stack di PNG ordinate come volume 3D SimpleITK."""
    frame_paths = _list_series_frame_paths(series_dir)
    slices = []
    for path in frame_paths:
        data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise FileNotFoundError(f"Impossibile leggere {path}")
        slices.append(data.astype(np.float32))
    volume = np.stack(slices, axis=0)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(pixel_spacing)
    return image


def load_dicom_series(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not names:
        raise ValueError(f"Nessun DICOM in {series_dir}")
    reader.SetFileNames(names)
    return reader.Execute()


def load_series_image(series_dir: Path) -> sitk.Image:
    """DICOM se presente, altrimenti stack PNG nella cartella serie."""
    try:
        return load_dicom_series(series_dir)
    except ValueError:
        return load_png_series_as_volume(series_dir)


def load_and_isotropize(image: sitk.Image, target_spacing=TARGET_SPACING) -> sitk.Image:
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(image)


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkNearestNeighbor,
    default_value=0,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(moving)


def create_multiview_mri_stack(
    axial_dir: Path = AXIAL_MRI_SERIES,
    sagittal_dirs: Sequence[Path] = SAGITTAL_MRI_SERIES,
    coronal_dir: Path = CORONAL_MRI_SERIES,
    target_spacing=TARGET_SPACING,
) -> Tuple[np.ndarray, sitk.Image]:
    """Restituisce stack MRI (3, Z, H, W) e immagine master assiale."""
    axial_master = load_and_isotropize(load_series_image(axial_dir), target_spacing)

    sag_aligned = [
        resample_to_reference(
            load_and_isotropize(load_series_image(s_dir), target_spacing),
            axial_master,
            interpolator=sitk.sitkBSpline,
            default_value=0.0,
        )
        for s_dir in sagittal_dirs
    ]
    sag_full = sag_aligned[0]
    for extra in sag_aligned[1:]:
        sag_full = sitk.Maximum(sag_full, extra)

    cor_aligned = resample_to_reference(
        load_and_isotropize(load_series_image(coronal_dir), target_spacing),
        axial_master,
        interpolator=sitk.sitkBSpline,
        default_value=0.0,
    )

    def z_score(img: sitk.Image) -> np.ndarray:
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    stack = np.stack(
        [z_score(axial_master), z_score(sag_full), z_score(cor_aligned)],
        axis=0,
    )
    return stack.astype(np.float32), axial_master


def _binary_mask_from_sitk(mask_img: sitk.Image) -> sitk.Image:
    binary = sitk.Cast(sitk.NotEqual(mask_img, 0), sitk.sitkUInt8)
    return sitk.BinaryDilate(binary, KIDNEY_MASK_DILATE_RADIUS)


def _list_kidney_mask_files(mask_dir: Path) -> List[Path]:
    if not mask_dir.exists():
        return []
    files = sorted(set(mask_dir.glob("*.nii")) | set(mask_dir.glob("*.nii.gz")), key=lambda p: p.name)
    return [path for path in files if path.is_file()]


def fuse_kidney_masks(axial_master: sitk.Image, mask_dirs: Dict[str, Path]) -> np.ndarray:
    """Unisce maschere rene AX/COR/SAG (tutti i NIfTI in ogni cartella) sul master assiale."""
    fused = sitk.Image(axial_master.GetSize(), sitk.sitkUInt8)
    fused.CopyInformation(axial_master)
    loaded = 0

    for orientation, mask_dir in mask_dirs.items():
        mask_files = _list_kidney_mask_files(mask_dir)
        if not mask_files:
            print(f"Avviso: nessuna maschera rene in {orientation} ({mask_dir})")
            continue
        for path in mask_files:
            native = sitk.ReadImage(str(path))
            aligned = resample_to_reference(native, axial_master)
            fused = sitk.Maximum(fused, _binary_mask_from_sitk(aligned))
            loaded += 1
            print(f"Maschera rene caricata ({orientation}): {path.name}")

    if loaded == 0:
        raise FileNotFoundError(
            "Nessuna maschera rene trovata in Kidney_mask/AX|COR|SAG. "
            "Inserisci file .nii o .nii.gz nelle rispettive cartelle."
        )

    return sitk.GetArrayFromImage(fused).astype(np.float32)


def parse_classmask_path(path: str) -> Tuple[str, str]:
    """Estrae (series_prefix, frame_id) da ..._SERIES_000034_classmask.png."""
    match = re.search(r"^(.*)_(\d+)_classmask(?:\.[^.]+)?$", Path(path).name)
    if not match:
        raise ValueError(f"Nome classmask non riconosciuto: {path}")
    return match.group(1), match.group(2)


def _frame_id_to_slice_index(series_dir: Path, frame_id: str) -> int:
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if names:
        stems = [Path(name).stem for name in names]
    else:
        stems = [path.stem for path in _list_series_frame_paths(series_dir)]
    matches = [idx for idx, stem in enumerate(stems) if stem.endswith(frame_id)]
    if not matches:
        raise ValueError(f"Frame {frame_id} non trovato in {series_dir}")
    if len(matches) > 1:
        raise ValueError(f"Frame {frame_id} ambiguo in {series_dir}: {matches}")
    return matches[0]


def embed_classmask_in_native_volume(classmask_path: str, series_dir: Path) -> sitk.Image:
    """Inserisce una classmask 2D nel volume 3D nativo della serie annotata."""
    series_prefix, frame_id = parse_classmask_path(classmask_path)
    if series_prefix not in CLASSMASK_SERIES_DIRS:
        raise ValueError(f"Serie {series_prefix} non in CLASSMASK_SERIES_DIRS")
    expected_dir = CLASSMASK_SERIES_DIRS[series_prefix]
    if expected_dir.resolve() != series_dir.resolve():
        raise ValueError(f"Mismatch serie: {classmask_path} -> {series_dir}")

    native_image = load_series_image(series_dir)
    slice_idx = _frame_id_to_slice_index(series_dir, frame_id)

    label_2d = classmask_to_label(read_classmask_image(classmask_path))
    volume = sitk.GetArrayFromImage(native_image)
    label_resized = cv2.resize(
        label_2d,
        (volume.shape[2], volume.shape[1]),
        interpolation=cv2.INTER_NEAREST,
    )

    label_volume = np.zeros_like(volume, dtype=np.uint8)
    label_volume[slice_idx] = label_resized

    label_image = sitk.GetImageFromArray(label_volume)
    label_image.CopyInformation(native_image)
    return label_image


def fuse_cyst_annotations(axial_master: sitk.Image, cyst_mask_dirs: Dict[str, Path]) -> np.ndarray:
    """Proietta tutte le classmask AX/COR/SAG sulla griglia assiale e fa unione."""
    fused = np.zeros(sitk.GetArrayFromImage(axial_master).shape, dtype=np.uint8)

    for orientation, mask_dir in cyst_mask_dirs.items():
        if not mask_dir.exists():
            print(f"Avviso: cartella annotazioni assente ({orientation}): {mask_dir}")
            continue

        classmasks = list_classmask_files(str(mask_dir))
        print(f"Annotazioni {orientation}: {len(classmasks)} classmask")
        for classmask_path in classmasks:
            series_prefix, _ = parse_classmask_path(classmask_path)
            series_dir = CLASSMASK_SERIES_DIRS[series_prefix]
            native_label = embed_classmask_in_native_volume(classmask_path, series_dir)
            aligned = resample_to_reference(native_label, axial_master)
            aligned_arr = sitk.GetArrayFromImage(aligned).astype(np.uint8)
            fused = np.maximum(fused, aligned_arr)

    return fused


def _resize_slice(channel_2d: np.ndarray, target_size=TARGET_SLICE_SIZE) -> np.ndarray:
    tensor = np.expand_dims(channel_2d.astype(np.float32), axis=-1)
    resized = tf.image.resize(tensor, list(target_size), method="bilinear").numpy()
    return resized[..., 0]


def _resize_label_slice(label_2d: np.ndarray, target_size=TARGET_SLICE_SIZE) -> np.ndarray:
    return cv2.resize(label_2d.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)


def build_volume_tensors(
    supervised_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, sitk.Image, np.ndarray]:
    """Costruisce X (N, H, W, 4), y (N, H, W), master image, indici Z usati."""
    mri_stack, axial_master = create_multiview_mri_stack()
    kidney_volume = fuse_kidney_masks(axial_master, KIDNEY_MASK_DIRS)
    cyst_volume = fuse_cyst_annotations(axial_master, CYST_MASK_DIRS)

    z_indices = np.arange(cyst_volume.shape[0])
    if supervised_only:
        z_indices = np.where(cyst_volume.sum(axis=(1, 2)) > 0)[0]
    if z_indices.size == 0:
        raise ValueError("Nessuna fetta con annotazioni cisti sulla griglia assiale.")

    x_slices: List[np.ndarray] = []
    y_slices: List[np.ndarray] = []
    for z in z_indices:
        channels = [
            _resize_slice(mri_stack[0, z]),
            _resize_slice(mri_stack[1, z]),
            _resize_slice(mri_stack[2, z]),
            _resize_slice(kidney_volume[z]),
        ]
        kidney_channel = (channels[3] > 0.5).astype(np.float32)
        x_slices.append(np.stack(channels[:3] + [kidney_channel], axis=-1))
        y_slices.append(_resize_label_slice(cyst_volume[z]))

    x = np.stack(x_slices, axis=0).astype(np.float32)
    y = np.stack(y_slices, axis=0).astype(np.uint8)
    print(f"Volume tensor: X {x.shape}, y {y.shape}, fette supervise Z={z_indices.tolist()}")
    return x, y, axial_master, z_indices


def build_full_volume_for_inference() -> Tuple[np.ndarray, sitk.Image, np.ndarray]:
    """Tensore X su tutte le fette assiali (per inferenza volume completo)."""
    mri_stack, axial_master = create_multiview_mri_stack()
    kidney_volume = fuse_kidney_masks(axial_master, KIDNEY_MASK_DIRS)

    x_slices: List[np.ndarray] = []
    for z in range(mri_stack.shape[1]):
        channels = [
            _resize_slice(mri_stack[0, z]),
            _resize_slice(mri_stack[1, z]),
            _resize_slice(mri_stack[2, z]),
            _resize_slice(kidney_volume[z]),
        ]
        kidney_channel = (channels[3] > 0.5).astype(np.float32)
        x_slices.append(np.stack(channels[:3] + [kidney_channel], axis=-1))

    return np.stack(x_slices, axis=0).astype(np.float32), axial_master, kidney_volume
