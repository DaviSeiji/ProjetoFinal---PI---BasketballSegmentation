import cv2
import numpy as np

#Binariza para garantir que a máscara é 0 ou 255
def binarizar_mask(mask, threshold=0):
    return ((mask > threshold).astype(np.uint8)) * 255

#Fecha buracos na máscara
def fechamento(mask, ksize=15):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#Seleciona o contorno mais elíptico baseado em uma pontuação, sendo necessário que o contorno tenha área e tamanho mínimos
def contorno_mais_eliptico(mask, min_area=200, min_bbox_size=120):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []

    for c in contornos:
        if len(c) < 20:
            continue

        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        side = max(w, h)
        if side < min_bbox_size:
            continue

        ellipse = cv2.fitEllipse(c)
        (cx, cy), (MA, ma), angle = ellipse

        area_elipse = np.pi * (MA/2) * (ma/2)
        score = min(area, area_elipse) / max(area, area_elipse)

        candidatos.append((score, c))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: x[0], reverse=True)
    return candidatos[0][1]

#Gera uma bounding box quadrada a partir do contorno fornecido
def fazer_bounding_box(contorno, img_shape):
    if contorno is None:
        return None

    x, y, w, h = cv2.boundingRect(contorno)
    side = max(w, h)

    cx = x + w // 2
    cy = y + h // 2

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(x1 + side, img_shape[1])
    y2 = min(y1 + side, img_shape[0])

    return x1, y1, x2, y2

#Gera uma máscara binária a partir da bounding box
def mask_from_bbox(box, shape):
    if box is None:
        return np.zeros(shape, dtype=np.uint8)

    x1, y1, x2, y2 = box
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

#Pós-processamento completo da máscara
def posproc_mask(mask):
    bin_mask = binarizar_mask(mask)
    closed = fechamento(bin_mask, ksize=15)
    best_contour = contorno_mais_eliptico(closed, min_area=50, min_bbox_size=30)
    box = fazer_bounding_box(best_contour, bin_mask.shape)
    final_mask = mask_from_bbox(box, bin_mask.shape)
    return final_mask
