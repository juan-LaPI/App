import torch
import torchvision as tv
import torch.nn as nn
import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------------------------
#                                    Resize3D
# ------------------------------------------------------------------------------------
class Resize3D:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32)
        resized_volume = nn.functional.interpolate(volume.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='trilinear')
        volumenFinal =  resized_volume.squeeze(0)
        
        return volumenFinal

# ------------------------------------------------------------------------------------
#                                 Inferencia RUBEN
# ------------------------------------------------------------------------------------
def CargarVolumen(root):
    volumeName = root
    sliceFiles = sorted(os.listdir(volumeName))
    volumeSlices = []
    for slice_file in sliceFiles:
        img_name = os.path.join(volumeName, slice_file)
        image = Image.open(img_name)
        volumeSlices.append(np.array(image))
    volume = np.stack(volumeSlices, axis=0)
    volume = torch.tensor(volume, dtype=torch.float32)
    volume = Resize3D(target_size=(64,112,112))(volume)
    volume = volume / torch.max(volume).item()
    volume = volume.repeat(3, 1, 1, 1).unsqueeze(0)
    return volume

def MViTV2S(weights):
    modelo = tv.models.video.mvit_v2_s()
    modelo.head = nn.Sequential(nn.Linear(in_features=768, out_features=512),
                                nn.Linear(512, 512),
                                nn.Linear(512, 4))
    stateDict = torch.load(weights, map_location=torch.device('cpu'),weights_only=True)
    modelo.load_state_dict(stateDict)
    return modelo

def uso_RUBEN(ruta_modelo, ruta_PNG):

	modelo = MViTV2S(ruta_modelo)
	volumen = CargarVolumen(ruta_PNG)

	modelo.eval()
	with torch.no_grad():
	    salida = modelo(volumen)

	eje = salida[0,1:].cpu().numpy()
	eje = eje / np.linalg.norm(eje)

	ang = float(salida[0][0])
	p_1, p_2, p_3 = float(eje[0]), float(eje[1]), float(eje[2])

	return ang, p_1, p_2, p_3

def uso_RUBEN_mult(ruta_modelo, rutas_PNG):
    parametros = []
    for ruta_PNG in rutas_PNG:
        prm = uso_RUBEN(ruta_modelo,ruta_PNG)
        parametros.append(prm)

    return parametros
# ------------------------------------------------------------------------------------
#                                 Inferencia YOLO
# ------------------------------------------------------------------------------------
def CargarVolumen_YOLO(root):
    volumeName = root
    sliceFiles = sorted(os.listdir(volumeName))
    volumeSlices = []
    for slice_file in sliceFiles:
        img_name = os.path.join(volumeName, slice_file)
        image = Image.open(img_name)
        volumeSlices.append(np.array(image))
    volume = np.stack(volumeSlices, axis=0)
    d, h, w = volume.shape
    volume = torch.tensor(volume, dtype=torch.float32)
    volume = Resize3D(target_size=(d,640,640))(volume)

    volume = volume.squeeze(0).unsqueeze(3).repeat(1, 1, 1, 3).numpy().astype(np.uint8)

    return volume

def uso_YOLO(ruta_modelo, rutas_PNG):

    vol_RGB, vol_masks = [], []
    indc, i = [], -1
    model = YOLO(ruta_modelo)

    for ruta_PNG in rutas_PNG:
        V = CargarVolumen_YOLO(ruta_PNG)
        results = model(V[0,:,:,:])
        r = results
        conf_arr = r[0].boxes.conf.cpu().numpy()
		
        if conf_arr.size and conf_arr.max() > 0.65:
            i += 1
            j = int(np.argmax(conf_arr))
            vol_RGB.append(cv2.cvtColor(r[0].plot(),cv2.COLOR_BGR2RGB))
            vol_masks.append(r[0].masks.data.cpu().numpy()[j,:,:])
            
            if r[0].boxes.cls.cpu().numpy()[0] == 0:
                indc.append(i)
            else:
                pass
        else:
            pass
            
    return np.array(vol_RGB), np.array(vol_masks), indc[0], indc[-1]
