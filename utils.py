import cv2
import numpy as np
from matplotlib import pyplot as plt

def img_cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def mostrar_imagem(titulo, imagen):
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocessar_imagem(img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)

    return img_thresh

def histograma(img):
    histograma = cv2.calcHist([img], [0], None, [256], [0, 256])

    return histograma

def alongamento_histograma(img):

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
        return img.astype(np.uint8)

    alongada = (img - min_val) * (255.0 / (max_val - min_val))
    alongada = np.clip(alongada, 0, 255).astype(np.uint8)

    return alongada

def equalizar_histograma(img):

    img_eq = cv2.equalizeHist(img)
    return img_eq

def clahe_histograma(img):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe


def medir_nitidez(img):
    laplaciano = cv2.Laplacian(img, cv2.CV_64F)

    return laplaciano.var()

def limiarizacao_adaptativa(img):

    img_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img_adapt

