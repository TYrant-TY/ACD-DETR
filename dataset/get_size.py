# coding: utf-8
import os
import cv2
from collections import Counter

def analyze_visdrone_sizes(img_folder):
    sizes = []
    for img_file in os.listdir(img_folder):
        if img_file.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(img_folder, img_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            sizes.append((w, h))
    
    size_counts = Counter(sizes)
    for size, count in size_counts.most_common():
        print(f"size: {size[0]}×{size[1]}: count: {count} (size[0]/size[1]: {size[0]/size[1]:.2f})")

analyze_visdrone_sizes("/root/ty/RTDETR-main/dataset/VisDrone/VisDrone2019-DET-train/images")