import SimpleITK as sitk
import matplotlib.pyplot as plt 
import numpy as np
import cv2

# ------------------------------------------------------------------------------------
#                                   Leer volumen DICOM
# ------------------------------------------------------------------------------------
def abrir_archivos_dicom(dicom_folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    volumen = reader.Execute()
    volumen_array = sitk.GetArrayFromImage(volumen)
	
    return volumen_array

def leer_archivos_dicom(dicom_folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    volumen = reader.Execute()

    return volumen

# ------------------------------------------------------------------------------------
#                             Remuestreo a espaciado definido
# ------------------------------------------------------------------------------------
def remuestrear_volumen(volumen, new_spacing, new_size=None, interpolador=sitk.sitkBSpline):#interpolador=sitk.sitkLinear):
    original_spacing = np.array(volumen.GetSpacing())
    original_size = np.array(volumen.GetSize())
    
    if new_size is None:
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolador)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputOrigin(volumen.GetOrigin())
    resampler.SetOutputDirection(volumen.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(volumen)

# ------------------------------------------------------------------------------------
#                      Rotación 3D alrededor de un eje arbitrario
# ------------------------------------------------------------------------------------
def aplicar_rotacion(volumen, axis, angulo_grados):
    axis = np.array(axis, dtype=float) * -1
    axis /= np.linalg.norm(axis)
    
    angulo_radianes = np.deg2rad(angulo_grados)
    transform = sitk.VersorRigid3DTransform()
    transform.SetRotation(axis.tolist(), angulo_radianes)
    
    centro = np.array(volumen.TransformContinuousIndexToPhysicalPoint(np.array(volumen.GetSize())/2.0))
    transform.SetCenter(centro.tolist())
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volumen)
    resampler.SetInterpolator(sitk.sitkBSpline) # Interpolador lineal 
    #resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(volumen)

# ------------------------------------------------------------------------------------
#                         Medir distancia real entre dos puntos
# ------------------------------------------------------------------------------------
def medir_distancia(volumen, p1, p2):
    p1_phys = np.array(volumen.TransformIndexToPhysicalPoint(p1))
    p2_phys = np.array(volumen.TransformIndexToPhysicalPoint(p2))
    dist = np.linalg.norm(p2_phys - p1_phys)
    return dist

# ------------------------------------------------------------------------------------
#                                  Pipeline Completo
# ------------------------------------------------------------------------------------
def process_dicom(angulo, axis, dicom_folder, medir=False):
    
    volumen_original = leer_archivos_dicom(dicom_folder)
    
    volumen_iso = remuestrear_volumen(volumen_original, new_spacing=(1.0, 1.0, 1.0))
    
    volumen_rot =  aplicar_rotacion(volumen_iso, axis, angulo)
    
    volumen_final = remuestrear_volumen(volumen_rot,
                                        new_spacing=volumen_original.GetSpacing(),
                                        new_size=volumen_original.GetSize())
    
    if medir:
        p1 = (20, 20, 20)
        p2 = (43, 43, 43)
        dist_orig = medir_distancia(volumen_original, p1, p2)
        dist_iso = medir_distancia(volumen_iso, p1, p2)
        dist_rot = medir_distancia(volumen_rot, p1, p2)
        dist_final = medir_distancia(volumen_final, p1, p2)

    volumen_np = sitk.GetArrayFromImage(volumen_final)
    spc = volumen_final.GetSpacing()
		
    return volumen_np, spc
