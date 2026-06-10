"""Percorsi per pipeline multi-vista (AX / COR / SAG) su griglia assiale master."""

from pathlib import Path

# Root dello studio (cartelle DICOM per serie)
STUDY_ROOT = Path("A/20240916-RM ABDOMEN(FP)")

# Serie RM usate per i 3 canali di input (resample su master assiale)
AXIAL_MRI_SERIES = STUDY_ROOT / "8001-AX T2 HASTE"
CORONAL_MRI_SERIES = STUDY_ROOT / "10001-COR T2 HASTE"
SAGITTAL_MRI_SERIES = [
    STUDY_ROOT / "11001-SAG T2 HASTE DERECHO",
    STUDY_ROOT / "12001-SAG T2 HASTE IZQUIERDO",
]

# Maschere rene 3D (NIfTI), cartelle per orientamento
KIDNEY_MASK_ROOT = Path("Kidney_mask")
KIDNEY_MASK_DIRS = {
    "AX": KIDNEY_MASK_ROOT / "AX",
    "COR": KIDNEY_MASK_ROOT / "COR",
    "SAG": KIDNEY_MASK_ROOT / "SAG",
}

# Annotazioni cisti 2D (PNG classmask), per orientamento
CYST_MASK_ROOT = Path("masks")
CYST_MASK_DIRS = {
    "AX": CYST_MASK_ROOT / "AX",
    "COR": CYST_MASK_ROOT / "COR",
    "SAG": CYST_MASK_ROOT / "SAG",
}

# Mappa prefisso nel nome file classmask -> cartella DICOM della serie annotata
CLASSMASK_SERIES_DIRS = {
    "6001-EX_AX_T2_FS": STUDY_ROOT / "6001-EX AX T2 FS",
    "10001-COR_T2_HASTE": STUDY_ROOT / "10001-COR T2 HASTE",
    "11001-SAG_T2_HASTE_DERECHO": STUDY_ROOT / "11001-SAG T2 HASTE DERECHO",
    "12001-SAG_T2_HASTE_IZQUIERDO": STUDY_ROOT / "12001-SAG T2 HASTE IZQUIERDO",
}

TARGET_SLICE_SIZE = (256, 256)
TARGET_SPACING = (1.0, 1.0, 1.0)
KIDNEY_MASK_DILATE_RADIUS = (2, 2, 2)
