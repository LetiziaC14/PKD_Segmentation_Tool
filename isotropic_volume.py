import SimpleITK as sitk
import numpy as np

def load_and_isotropize_dicom(dicom_directory, target_spacing=(1.0, 1.0, 1.0)):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_directory)
    # Carica la prima serie trovata nella cartella
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory, series_IDs[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Resampling isotropo (B-Spline)
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [int(round(osz * osp / tsp)) for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)

def align_to_master(fixed_image, moving_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(moving_image)

def create_multi_view_stack(axial_dir, sagittal_dirs, coronal_dir):
    # 1. Carica e isotropizza il Master Assiale
    axial_master = load_and_isotropize_dicom(axial_dir)
    
    # 2. Carica e allinea i due Sagittali (Right + Left) e uniscili
    sag_aligned = []
    for s_dir in sagittal_dirs:
        raw = load_and_isotropize_dicom(s_dir)
        sag_aligned.append(align_to_master(axial_master, raw))
    
    # Stitching dei Sagittali
    sag_full = sitk.Maximum(sag_aligned[0], sag_aligned[1])
    
    # 3. Carica e allinea l'unico Coronale
    cor_raw = load_and_isotropize_dicom(coronal_dir)
    cor_aligned = align_to_master(axial_master, cor_raw)
    
    # 4. Creazione tensore (3, Z, H, W) con normalizzazione Z-score
    def z_score(img):
        arr = sitk.GetArrayFromImage(img)
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    stack = np.stack([
        z_score(axial_master),
        z_score(sag_full),
        z_score(cor_aligned)
    ], axis=0)
    
    return stack.astype(np.float32), axial_master

if __name__ == "__main__":
    from pathlib import Path

    study_root = Path("A/20240916-RM ABDOMEN(FP)")
    axial_path = study_root / "8001-AX T2 HASTE"
    sagittal_paths = [
        study_root / "11001-SAG T2 HASTE DERECHO",
        study_root / "12001-SAG T2 HASTE IZQUIERDO",
    ]
    coronal_path = study_root / "10001-COR T2 HASTE"

    input_stack, axial_master = create_multi_view_stack(
        str(axial_path),
        [str(p) for p in sagittal_paths],
        str(coronal_path),
    )
    print(f"Tensore multi-view creato con shape: {input_stack.shape}")