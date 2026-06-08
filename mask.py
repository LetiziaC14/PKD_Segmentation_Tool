import SimpleITK as sitk
import numpy as np
from isotropic_volume import input_stack, axial_master

def prepare_kidney_mask(coronal_mask_path, axial_master):
    """
    Carica la maschera coronale, la allinea al Master Assiale, 
    la dilata leggermente e la prepara per il tensore.
    """
    # 1. Carica la maschera originale
    mask_img = sitk.ReadImage(coronal_mask_path)
    
    # 2. Allineamento rigoroso al Master Assiale
    # Usiamo NearestNeighbor per mantenere i valori 0/1 (binari) intatti
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(axial_master)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    aligned_mask = resampler.Execute(mask_img)
    
    # 3. Converti eventuali etichette multiple in una maschera binaria
    #    (tutti i valori non-zero diventano 1), quindi dilata
    bin_mask = sitk.NotEqual(aligned_mask, 0)
    bin_mask = sitk.Cast(bin_mask, sitk.sitkUInt8)

    # Dilatazione morfologica (opzionale ma consigliata)
    # Crea un raggio di 2-3 voxel per includere cisti in prossimità del rene
    dilated_mask = sitk.BinaryDilate(bin_mask, [2, 2, 2])

    return dilated_mask

def final_pipeline_integration(input_stack, axial_master, coronal_mask_path):
    # Genera la maschera 3D allineata e dilatata
    kidney_mask_3d = prepare_kidney_mask(coronal_mask_path, axial_master)
    
    # Converte in array NumPy
    mask_array = sitk.GetArrayFromImage(kidney_mask_3d).astype(np.float32)
    
    # Verifica dimensionale (deve essere Z, H, W)
    # Assicurati che mask_array sia (280, 370, 400) come il tuo stack
    if mask_array.shape != input_stack.shape[1:]:
        raise ValueError(f"Dimensione maschera {mask_array.shape} non coincide con stack {input_stack.shape[1:]}")
    
    # Aggiungi dimensione canale: (1, 280, 370, 400)
    mask_channel = mask_array[np.newaxis, :, :, :]
    
    # Concatenazione finale per ottenere (4, 280, 370, 400)
    final_input_tensor = np.concatenate([input_stack, mask_channel], axis=0)
    
    return final_input_tensor.astype(np.float32)


def save_mask_nifti(mask_image, output_path):
    """Salva la maschera 3D allineata come file NIfTI per Slicer."""
    sitk.WriteImage(mask_image, output_path)
    print(f"Maschera salvata come NIfTI: {output_path}")


def save_volume_nifti(volume_image, output_path):
    """Salva un volume SimpleITK (es. `axial_master`) come NIfTI."""
    sitk.WriteImage(volume_image, output_path)
    print(f"Volume salvato come NIfTI: {output_path}")


# --- EXAMPLE EXECUTION ---
# Uses input_stack and axial_master imported from isotropic_volume.py
# and applies the coronal mask to produce the final input tensor.

coronal_mask_path = "Kidney_mask/10001 COR T2 HASTE.nii.gz"
kidney_mask_3d = prepare_kidney_mask(coronal_mask_path, axial_master)
save_mask_nifti(kidney_mask_3d, "Kidney_mask/kidney_mask_3d.nii.gz")

# Salva anche il volume RM assiale (axial_master) come NIfTI per Slicer
save_volume_nifti(axial_master, "Dicom_Torra/axial_master.nii.gz")

final_tensor = final_pipeline_integration(input_stack, axial_master, coronal_mask_path)
print(f"Final tensor ready for U-Net: {final_tensor.shape}")
