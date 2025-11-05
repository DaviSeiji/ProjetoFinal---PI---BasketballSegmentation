import cv2
import numpy as np
import matplotlib.pyplot as plt

caminho = 'data/images/imagem9.jpg'
img = cv2.imread(caminho)
if img is None:
    raise ValueError("Erro ao carregar a imagem!")

# Converter para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Faixa de cor para laranja escuro
lower_orange = np.array([5, 120, 80])
upper_orange = np.array([15, 255, 255])
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Limpeza morfológica
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
mask = cv2.medianBlur(mask, 7)

# Encontrar contornos
contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contornos:
    for c in contornos:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue

        circularidade = 4 * np.pi * (area / (perimetro ** 2))
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)

        # Critérios de detecção mais flexíveis
        if 500 < area < 30000 and 0.3 < circularidade < 1.3 and 0.7 < aspect_ratio < 1.3:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Bola", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            print(f"Bola detectada: área={area:.1f}, circ={circularidade:.2f}, ar={aspect_ratio:.2f}")
else:
    print("Nenhum contorno encontrado.")

# Mostrar resultados
fig, axs = plt.subplots(1, 3, figsize=(12,5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title('Detecção da Bola')
axs[1].imshow(mask, cmap='gray'); axs[1].set_title('Máscara Laranja Escuro')
axs[2].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)); axs[2].set_title('Imagem HSV')
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()
