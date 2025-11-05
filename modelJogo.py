import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_e_preparar_imagem(caminho, escala=0.5, margem_crop=0.15):

    # Carregar imagem
    img = cv2.imread(caminho)
    if img is None:
        raise ValueError(f"Erro ao carregar a imagem: {caminho}")

    # Redimensionar mantendo proporção
    img_resized = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    # Recortar imagem para focar na região central, removendo partes que podem atrapalhar
    h, w = img_resized.shape[:2]
    y1, y2 = int(h * margem_crop), int(h * (1 - margem_crop))
    x1, x2 = int(w * margem_crop), int(w * (1 - margem_crop))
    img_crop = img_resized[y1:y2, x1:x2]

    return img_crop


def corrigir_white_balance(img):

    b, g, r = cv2.split(img)
    
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    mean_gray = (mean_r + mean_g + mean_b) / 3

    kr, kg, kb = mean_gray / mean_r, mean_gray / mean_g, mean_gray / mean_b

    r_corr = np.clip(r * kr, 0, 255).astype(np.uint8)
    g_corr = np.clip(g * kg, 0, 255).astype(np.uint8)
    b_corr = np.clip(b * kb, 0, 255).astype(np.uint8)

    img_corrigida = cv2.merge([b_corr, g_corr, r_corr])
    return img_corrigida


def reduzir_ruido(img):

    img_denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img_denoised



if __name__ == "__main__":

    qtd_imagens = 3

    for i in range(qtd_imagens):
        caminho_imagem = f'data/imagesJogo/imagem{i+1}.jpg'
        imagem = carregar_e_preparar_imagem(caminho_imagem)

        imagem_corrigida = corrigir_white_balance(imagem)

        imagem_menor_ruido = reduzir_ruido(imagem_corrigida)

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        axs[0,0].imshow(imagem)
        axs[0,0].set_title('Original')

        axs[0,1].imshow(imagem_corrigida)
        axs[0,1].set_title('White Balance Corrigido')

        axs[0,2].imshow(imagem_menor_ruido)
        axs[0,2].set_title('Redução de Ruído')

        plt.tight_layout()
        plt.show()
