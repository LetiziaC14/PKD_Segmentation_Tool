import SimpleITK as sitk
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------
# 1. CO-REGISTRAZIONE INVERSA (Resampling su griglia nativa)
# ---------------------------------------------------------
def resample_mask_to_native(native_mri_sitk, mask_3d_nifti_path):
    """
    Proietta il volume 3D della maschera (.nii.gz) sulla griglia 
    spaziale esatta della serie RM nativa (DICOM).
    """
    # Carica la maschera 3D
    mask_3d_sitk = sitk.ReadImage(mask_3d_nifti_path)
    
    # Configura il resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(native_mri_sitk)
    # TASSATIVO: NearestNeighbor per mantenere la maschera binaria (0 o 1)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    
    # Esegue l'allineamento
    native_mask_sitk = resampler.Execute(mask_3d_sitk)
    return native_mask_sitk

# ---------------------------------------------------------
# 2. & 3. SLICING SINCRONIZZATO E PREPARAZIONE DELL'INPUT
# ---------------------------------------------------------
def prepare_2d_inputs_from_native(native_mri_sitk, native_mask_sitk):
    """
    Estrae le fette, esegue il ridimensionamento a 256x256 e applica 
    la doppia normalizzazione richiesta dallo studio.
    """
    # Converti in array NumPy. 
    # Shape risultante: (Slice, Altezza, Larghezza) ovvero (Z, Y, X)
    mri_arr = sitk.GetArrayFromImage(native_mri_sitk).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(native_mask_sitk).astype(np.float32)
    
    # Normalizzazione al 95° percentile per l'intero volume RM
    p95 = np.percentile(mri_arr, 95)
    if p95 > 0:
        mri_arr = mri_arr / p95
        
    # Standard scalar normalization (zero mean, unit std)
    mean_val = np.mean(mri_arr)
    std_val = np.std(mri_arr)
    if std_val > 0:
        mri_arr = (mri_arr - mean_val) / std_val

    num_slices = mri_arr.shape[0]
    processed_slices = []
    
    for i in range(num_slices):
        # Slicing sincronizzato: estrazione della i-esima fetta
        mr_slice = mri_arr[i, :, :]
        mask_slice = mask_arr[i, :, :]
        
        # Aggiungi la dimensione del canale per il ridimensionamento
        mr_slice = np.expand_dims(mr_slice, axis=-1)
        mask_slice = np.expand_dims(mask_slice, axis=-1)
        
        # Ridimensionamento a 256x256 con metodi separati
        # Interpolazione bicubica/bilineare per RM, Nearest Neighbor per maschera[cite: 1]
        mr_resized = tf.image.resize(mr_slice, [256, 256], method='bilinear').numpy()
        mask_resized = tf.image.resize(mask_slice, [256, 256], method='nearest').numpy()
        
        # Binarizzazione di sicurezza sulla maschera dopo il resize
        mask_resized = (mask_resized > 0.5).astype(np.float32)
        
        # Concatenazione finale per formare l'input a due canali[cite: 1]
        # Shape di combined_input: (256, 256, 2)
        combined_input = np.concatenate([mr_resized, mask_resized], axis=-1)
        
        processed_slices.append(combined_input)
        
    # Impila tutte le fette in un unico tensore (Batch)
    # Shape finale: (N, 256, 256, 2)
    return np.stack(processed_slices, axis=0)

# ---------------------------------------------------------
# ESECUZIONE (Esempio su una singola vista nativa)
# ---------------------------------------------------------
def process_single_view(dicom_dir, mask_nifti_path):
    # 1. Caricamento della serie DICOM nativa
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    native_mri_sitk = reader.Execute()
    
    # 2. Co-registrazione inversa della maschera sul piano nativo
    native_mask_sitk = resample_mask_to_native(native_mri_sitk, mask_nifti_path)
    
    # 3. Preparazione del tensore finale
    input_tensor = prepare_2d_inputs_from_native(native_mri_sitk, native_mask_sitk)
    
    return input_tensor

def main():
    dicom_coronal_path = "Dicom_Torra/10001-COR T2 HASTE"
    mask_nifti_path = "Kidney_mask/kidney_mask_3d.nii.gz"
    tensor_coronal = process_single_view(dicom_coronal_path, mask_nifti_path)
    print(f"Shape tensore pronto per la U-Net: {tensor_coronal.shape}")


if __name__ == '__main__':
    main()