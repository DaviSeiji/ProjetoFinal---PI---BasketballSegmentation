import cv2
import os
import numpy as np

path_img = "data/img"
path_masks = "data/real_masks"
save_path = "results/real_masks"

os.makedirs(save_path, exist_ok=True)

# Lista das imagens JPG
images = [f for f in os.listdir(path_img) if f.lower().endswith(".jpg")]

if not images:
    raise FileNotFoundError("Nenhuma imagem encontrada em data/img")


def get_bbox_from_mask(mask):
    """Retorna bounding box a partir da máscara."""
    mask_bin = (mask > 127).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask_dil = cv2.dilate(mask_bin, kernel, iterations=2)

    contornos, _ = cv2.findContours(mask_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contornos]

    return bboxes

for img_name in images:
    name = os.path.splitext(img_name)[0]

    img_path = os.path.join(path_img, img_name)
    mask_path = os.path.join(path_masks, name + ".png")

    if not os.path.exists(mask_path):
        print(f"⚠️ Máscara não encontrada para {img_name}")
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"❌ Erro ao ler {img_path} ou {mask_path}")
        continue

    bboxes = get_bbox_from_mask(mask)

    img_bbox = img.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)

    save_file = os.path.join(save_path, name + "_real.jpg")
    cv2.imwrite(save_file, img_bbox)
    print(f"💾 Salvo: {save_file}")
