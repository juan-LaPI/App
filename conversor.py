import numpy as np
from PIL import Image
import SimpleITK as sitk
import tempfile
import os
import uuid
import zipfile
import io

# ------------------------------------------------------------------------------------
#                                    Carpeta PNG
# ------------------------------------------------------------------------------------
def img_slc(Vol,img_mode,SliceNum):

    if img_mode == 0:
        Vol_raw = Vol[SliceNum,:,:]
        if np.max(Vol_raw) != np.min(Vol_raw):
            Vol_norm = (Vol_raw-np.min(Vol_raw))/(np.max(Vol_raw)-np.min(Vol_raw))
            Vol_img = (Vol_norm*255).astype(np.uint8)
            img_user = Image.fromarray(Vol_img,mode='L')
        else:
            img_user = Image.fromarray(Vol_raw,mode='L')

    else:
        Vol_raw = Vol[SliceNum,:,:,:]
        Vol_norm = (Vol_raw-np.min(Vol_raw))/(np.max(Vol_raw)-np.min(Vol_raw))
        Vol_img = (Vol_norm*255).astype(np.uint8)
        img_user = Image.fromarray(Vol_img,mode='RGB')  

    return img_user

def carpetaPNG(Vol,img_mode):
    Vol = np.array(Vol)

    temp_png = tempfile.mkdtemp()
    png_paths = []
    
    for i in range(Vol.shape[0]):
        img_png = img_slc(Vol,img_mode,i)
        path = os.path.join(temp_png, f'slice_{i:03d}.png')
        img_png.save(path)
        png_paths.append(path)

    return temp_png

def carpetaPNG_paths(Vol,img_mode):
    Vol = np.array(Vol)

    temp_png = tempfile.mkdtemp()
    png_paths = []
    
    for i in range(Vol.shape[0]):
        img_png = img_slc(Vol,img_mode,i)
        path = os.path.join(temp_png, f'slice_{i:03d}.png')
        img_png.save(path)
        png_paths.append(path)

    return temp_png, png_paths

# ------------------------------------------------------------------------------------
#                                    Carpeta DCM
# ------------------------------------------------------------------------------------
def uid():
    return '2.25.' + str(int(uuid.uuid4()))

def carpetaDCM(volume_np, spacing=(1, 1, 1),
                           modality='CT',
                           patient_name='Paciente',
                           patient_id='0000'):
    
    temp_dir_dicom = tempfile.mkdtemp()
    dicom_paths = []

    volume_np = volume_np.astype(np.int16)

    img3d = sitk.GetImageFromArray(volume_np)
    img3d.SetSpacing(spacing)

    study_uid = uid()
    series_uid = uid()

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    array = sitk.GetArrayFromImage(img3d)

    for z in range(array.shape[0]):
        slice_np = array[z, :, :]

        slice_img = sitk.GetImageFromArray(slice_np[None, :, :])
        slice_img.SetSpacing(spacing)

        slice_img.SetMetaData('0008|0060', modality)
        slice_img.SetMetaData('0010|0010', patient_name)
        slice_img.SetMetaData('0010|0020', patient_id)

        slice_img.SetMetaData('0020|000D', study_uid)
        slice_img.SetMetaData('0020|000E', series_uid)
        slice_img.SetMetaData('0008|0018', uid())

        slice_img.SetMetaData('0020|0013', str(z + 1))
        slice_img.SetMetaData('0020|0032', f'0\\0\\{z*spacing[2]}')
        slice_img.SetMetaData('0020|0037', '1\\0\\0\\0\\1\\0')

        slice_img.SetMetaData('0018|0050', str(spacing[2]))

        out_path = os.path.join(temp_dir_dicom, f'slice_{z:03d}.dcm')
        writer.SetFileName(out_path)
        writer.Execute(slice_img)

        dicom_paths.append(out_path)

    return temp_dir_dicom

# ------------------------------------------------------------------------------------
#                                   Descarga PNG
# ------------------------------------------------------------------------------------
def descargaPNG(folder_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                path = os.path.join(root, file)
                arcname = os.path.relpath(path, folder_path)
                zipf.write(path, arcname)

    buffer.seek(0)
    return buffer
