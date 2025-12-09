import os
import json
import cv2
import numpy as np

# Pastas principais
base_path = "data"
output_base = "real_masks"

# Percorrer cada split (train, valid, test)
for split in ["train", "valid", "test"]:
    split_path = os.path.join(base_path, split)
    images_path = os.path.join(split_path, "images")
    ann_path = os.path.join(split_path, "_annotations.coco.json")
    masks_path = os.path.join(output_base, split)

    os.makedirs(masks_path, exist_ok=True)

    # Verifica se o arquivo de anotações existe
    if not os.path.exists(ann_path):
        print(f"[!] Anotações não encontradas para {split}, pulando.")
        continue

    print(f"[+] Gerando máscaras para: {split}")

    # Carrega anotações
    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Gera máscara para cada imagem
    for img_info in data['images']:
        img_name = img_info['file_name']
        img_id = img_info['id']
        h, w = img_info['height'], img_info['width']

        # Cria máscara vazia
        mask = np.zeros((h, w), dtype=np.uint8)

        # Adiciona todas as bounding boxes dessa imagem
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                x, y, bw, bh = map(int, ann['bbox'])
                cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

        # Salva a máscara
        mask_name = img_name.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
        mask_path = os.path.join(masks_path, mask_name)
        cv2.imwrite(mask_path, mask)

print("✅ Máscaras geradas com sucesso!")
