import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

#Segmentação baseada em HSV para a cor laranja
def hsv_laranja(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

#Segmentação baseada em Hough para detecção de círculos
def hough_mask(mask):
    blur = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=60,
        param1=80, param2=18, minRadius=25, maxRadius=200
    )

    mask_circle = np.zeros_like(mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(mask_circle, (x, y), r, 255, -1)

    return mask_circle

#Segmentação usando Otsu e Multi-Otsu
def otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def multiotsu(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresholds = threshold_multiotsu(gray, classes=3)
    regions = np.digitize(gray, bins=thresholds)

    mask = np.where(regions == 2, 255, 0).astype(np.uint8)
    return mask

#Segmentação usando Lab, focando na região perto da cor laranja
def lab_laranja(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, _, b = cv2.split(lab)
    _, mask = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

#Transformando em HSV e aplicando Otsu no canal V
def hsv_otsu(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

#Mesmo com Multi-Otsu
def hsv_multiotsu(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return multiotsu(v)

#Aplicando Lab com Hough
def lab_hough(img):
    return hough_mask(lab_laranja(img))

#Lab com Otsu
def lab_otsu(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2]
    _, mask = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

#Lab com Multi-Otsu
def lab_multiotsu(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2]
    return multiotsu(b)

#Gray com Otsu
def gray_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

#Gray com Multi-Otsu
def gray_multiotsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return multiotsu(gray)

#Gray com Hough
def gray_hough(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hough_mask(gray)

#Gray com Otsu seguido de Hough
def gray_otsu_hough(img):
    mask_otsu = gray_otsu(img)
    return hough_mask(mask_otsu)

