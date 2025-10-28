import cv2
import numpy as np
from matplotlib import pyplot as plt


def mostrar_imagem(titulo, imagen):
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocessar_imagem(img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)

    return img_thresh


def main():

    qtd_imagens = 3

    for i in range(qtd_imagens):

        caminho_imagem = f'data/images/imagem{i+1}.jpg'

        imagem = cv2.imread(caminho_imagem)

        if imagem is None:
            print(f'Erro ao carregar a imagem: {caminho_imagem}')
            continue

        mostrar_imagem(f'Imagem {i}', imagem)
        imagem_processada = preprocessar_imagem(imagem)
        mostrar_imagem(f'Imagem Processada {i}', imagem_processada)


if __name__ == "__main__":
    main()