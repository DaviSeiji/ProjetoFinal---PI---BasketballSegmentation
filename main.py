import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import *
from preproc.preproc_func import *
from segmentacao.segmentacao_func import *
from posproc.posproc_func import *

#Segmentadores testados baseados em HSV, Lab e Gray
segmentadores_hsv = {
    "HSV + Filtro": hsv_laranja,
    "HSV + Hough": lambda img: hough_mask(hsv_laranja(img)),
    "HSV + Otsu": hsv_otsu,
    "HSV + Multi-Otsu": hsv_multiotsu
}

segmentadores_lab = {
    "Lab + Filtro": lab_laranja,
    "Lab + Hough": lab_hough,
    "Lab + Otsu": lab_otsu,
    "Lab + Multi-Otsu": lab_multiotsu
}

segmentadores_gray = {
    "Gray + Otsu": gray_otsu,
    "Gray + Multi-Otsu": gray_multiotsu,
    "Gray + Hough": gray_hough,
    "Gray + Otsu + Hough": gray_otsu_hough
}

#Diferentes pré-processamentos testados
preprocessamentos = {
    "Original": lambda x: x,
    "CLAHE": clahe,
    "Normalize": normalize_image,
    "Bilateral": bilateral_filter,
    "Sharpen": sharpen_image,
    "CLAHE + Bilateral": lambda x: bilateral_filter(clahe(x)),
    "CLAHE + Sharpen": lambda x: sharpen_image(clahe(x)),
    "Normalize + Sharpen": lambda x: sharpen_image(normalize_image(x))
}

#Caminhos das pastas                                                                                    
path_img = "data/img"
path_mask = "data/real_masks"
path_out = "results"

os.makedirs(path_out, exist_ok=True)

images = [f for f in os.listdir(path_img) if f.lower().endswith(".jpg")]

if not images:
    raise FileNotFoundError("Nenhuma imagem encontrada em data/img")

resultados = {}
melhor_por_imagem = {}

#Função para salvar resultados
def registrar(nome, valor):
    if nome not in resultados:
        resultados[nome] = []
    resultados[nome].append(valor)

#Passagem pela base de imagens
for img_name in images:
    img = cv2.imread(os.path.join(path_img, img_name))
    mask_true_path = os.path.join(path_mask, img_name.replace(".jpg", ".png"))

    if not os.path.exists(mask_true_path):
        print(f"Máscara real não encontrada para {img_name}, pulando Dice.")
        continue

    mask_true = cv2.imread(mask_true_path, cv2.IMREAD_GRAYSCALE)

    best_dice = -1
    best_method = None
    best_mask = None

    for prep_name, prep in preprocessamentos.items():
        img_p = prep(img)
        for nome, func in segmentadores_hsv.items():
            mask_pred = posproc_mask(func(img_p))
            dice = dice_coefficient(mask_pred, mask_true)
            registrar(f"HSV | {prep_name} | {nome}", dice)
            if dice > best_dice:
                best_dice = dice
                best_method = f"HSV | {prep_name} | {nome}"
                best_mask = mask_pred

    for prep_name, prep in preprocessamentos.items():
        img_p = prep(img)
        for nome, func in segmentadores_lab.items():
            mask_pred = posproc_mask(func(img_p))
            dice = dice_coefficient(mask_pred, mask_true)
            registrar(f"LAB | {prep_name} | {nome}", dice)
            if dice > best_dice:
                best_dice = dice
                best_method = f"LAB | {prep_name} | {nome}"
                best_mask = mask_pred

    for prep_name, prep in preprocessamentos.items():
        img_p = prep(img)
        for nome, func in segmentadores_gray.items():
            mask_pred = posproc_mask(func(img_p))
            dice = dice_coefficient(mask_pred, mask_true)
            registrar(f"GRAY | {prep_name} | {nome}", dice)
            if dice > best_dice:
                best_dice = dice
                best_method = f"GRAY | {prep_name} | {nome}"
                best_mask = mask_pred

    base = os.path.splitext(img_name)[0]
    img_bbox = desenhar_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), best_mask)
    cv2.imwrite(os.path.join(path_out, f"bbox_{base}.png"), cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))

    print(f"{img_name}: melhor Dice = {best_dice:.4f} ({best_method})")

ranking = []
for metodo, valores in resultados.items():
    media = sum(valores) / len(valores)
    ranking.append((metodo, media))

ranking.sort(key=lambda x: x[1], reverse=True)

print("\nMÉDIA FINAL DOS MÉTODOS:\n")
for metodo, media in ranking:
    print(f"{metodo:<50} → MÉDIA DICE = {media:.4f}")
