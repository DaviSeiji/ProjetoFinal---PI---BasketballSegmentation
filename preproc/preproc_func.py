import cv2
import numpy as np

def blur_image(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def sharpen_image(image, strength=1.0):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + strength, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def normalize_image(image, alpha=0, beta=255):
    return cv2.normalize(image, None, alpha, beta, cv2.NORM_MINMAX)

def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return c.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = c.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
