import cv2
import numpy as np
import os

MIN_AREA = 3500
MIN_CIRC = 0.55

def dice_coefficient(mask_pred, mask_true):

    if mask_pred is None or mask_true is None:
        return 0.0
    mask_pred = (mask_pred > 0).astype(np.uint8)
    mask_true = (mask_true > 0).astype(np.uint8)
    intersection = np.sum(mask_pred * mask_true)
    return (2. * intersection) / (np.sum(mask_pred) + np.sum(mask_true) + 1e-6)

def circularidade(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per == 0:
        return 0
    return 4 * np.pi * area / (per ** 2)


def get_bbox(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contornos:
        if cv2.contourArea(c) < MIN_AREA: continue
        if circularidade(c) < MIN_CIRC: continue
        bboxes.append(cv2.boundingRect(c))
    return bboxes

def desenhar_bbox(img, mask):
    if mask is None:
        return img

    ys, xs = np.where(mask > 0)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    img_bbox = img.copy()
    img_bbox = cv2.rectangle(
        img_bbox,
        (xmin, ymin),
        (xmax, ymax),
        color=(255, 0, 0),
        thickness=3
    )

    return img_bbox

