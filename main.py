import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils as ut


def equalizando_histograma(imagem_cinza):
    std_contraste = imagem_cinza.std()
    min_val, max_val = imagem_cinza.min(), imagem_cinza.max()
    amplitude = max_val - min_val

    if std_contraste < 40 and amplitude < 100:
        print(f'Contraste muito baixo ({std_contraste:.1f}), amplitude {amplitude} → CLAHE.')
        equalizada = ut.clahe_histograma(imagem_cinza)

    elif std_contraste < 70 and amplitude < 200:
        print(f'Contraste moderado ({std_contraste:.1f}), amplitude {amplitude} → equalização global.')
        equalizada = ut.equalizar_histograma(imagem_cinza)

    elif std_contraste < 70:
        print(f'Contraste moderado, mas amplitude boa ({amplitude}) → alongamento linear.')
        equalizada = ut.alongamento_histograma(imagem_cinza)

    else:
        print(f'Contraste adequado ({std_contraste:.1f}) → sem processamento.')
        equalizada = imagem_cinza

    return equalizada


def borrando_imagem(equalizada):
    nitidez = ut.medir_nitidez(equalizada)

    if nitidez < 50:
        print(f"Imagem muito ruidosa → aplicando borramento forte.")
        borrada = cv2.GaussianBlur(equalizada, (9,9), 0)
    elif nitidez < 150:
        print(f"Imagem levemente ruidosa → borramento médio.")
        borrada = cv2.GaussianBlur(equalizada, (5,5), 0)
    else:
        print(f"Imagem nítida → borramento leve.")
        borrada = cv2.GaussianBlur(equalizada, (3,3), 0)

    return borrada


def main():
    qtd_imagens = 20

    for i in range(qtd_imagens):
        caminho_imagem = f'data/images/imagem{i+1}.jpg'
        imagem = cv2.imread(caminho_imagem)

        if imagem is None:
            print(f'Erro ao carregar a imagem: {caminho_imagem}')
            continue

        imagem_cinza = ut.img_cinza(imagem)
        histograma = ut.histograma(imagem_cinza)

        equalizada = equalizando_histograma(imagem_cinza)
        histograma_eq = ut.histograma(equalizada)

        borrada = borrando_imagem(equalizada)
        limiarizada = ut.limiarizacao_adaptativa(borrada)

        # Encontra contornos
        contornos, _ = cv2.findContours(limiarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"\nImagem {i+1}: {len(contornos)} contornos encontrados.")

        # Desenhar círculos nas bolas detectadas
        for c in contornos:
            area = cv2.contourArea(c)
            perimetro = cv2.arcLength(c, True)

            if perimetro == 0:
                continue

            circularidade = 4 * np.pi * (area / (perimetro**2))

            # Critérios para considerar uma bola
            if area > 500 and circularidade > 0.6:
                (x, y), raio = cv2.minEnclosingCircle(c)
                cv2.circle(imagem, (int(x), int(y)), int(raio), (0, 255, 0), 2)
                print(f"→ Bola detectada: área={area:.1f}, circ={circularidade:.2f}")

        # Plot das etapas
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        axs[0,0].imshow(imagem_cinza, cmap='gray')
        axs[0,0].set_title('Original')

        axs[0,1].plot(histograma)
        axs[0,1].set_title('Histograma Original')

        axs[0,2].imshow(equalizada, cmap='gray')
        axs[0,2].set_title('Equalizada / Processada')

        axs[1,0].imshow(borrada, cmap='gray')
        axs[1,0].set_title('Borrada')

        axs[1,1].imshow(limiarizada, cmap='gray')
        axs[1,1].set_title('Limiarização Adaptativa')

        axs[1,2].imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        axs[1,2].set_title('Bolas Detectadas')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
