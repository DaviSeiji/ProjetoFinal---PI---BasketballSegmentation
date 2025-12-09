import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import *
from preproc.preproc_func import *
from segmentacao.segmentacao_func import *
from posproc.posproc_func import *

#Teste Melhor resultado CLAHE | HSV + Filtro


path_img = "data/img"
path_mask = "data/real_masks"

images = [f for f in os.listdir(path_img) if f.lower().endswith(".jpg")]

if not images:
    raise FileNotFoundError("Nenhuma imagem encontrada em data/img")

resultados = {}
melhor_por_imagem = {}

for img_name in images:
    img_path = os.path.join(path_img, img_name)
    mask_path = os.path.join(path_mask, img_name.replace(".jpg", ".png"))

    img = cv2.imread(img_path)

    real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = posproc_mask(hsv_laranja(clahe(img)))

    if mask is None:
        resultados[img_name] = 0.0
        print(f"{img_name}: Dice = 0.0000")
        continue

    dice = dice_coefficient(real_mask, mask)

    resultados[img_name] = dice

    print(f"{img_name}: Dice = {dice:.4f}")

