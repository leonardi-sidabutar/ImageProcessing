import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Konversi ke biner
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Operasi morfologi
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Segmentasi menggunakan mask
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=morph)
    
    # Buat gambar dengan latar belakang transparan
    alpha = np.where(morph > 0, 255, 0).astype(np.uint8)
    image_rgba = np.dstack((image_rgb, alpha))
    
    # Buat gambar dengan latar belakang hitam dan hanya objek yang tampil
    segmented_black_bg = np.zeros_like(image_rgb)
    segmented_black_bg[morph > 0] = image_rgb[morph > 0]
    
    # Plot hasil
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title("RGB")
    axs[0, 0].axis("off")
    
    axs[0, 1].imshow(gray, cmap="gray")
    axs[0, 1].set_title("Grayscale")
    axs[0, 1].axis("off")
    
    axs[0, 2].imshow(binary, cmap="gray")
    axs[0, 2].set_title("Binary")
    axs[0, 2].axis("off")
    
    axs[1, 1].imshow(morph, cmap="gray")
    axs[1, 1].set_title("Morfologi")
    axs[1, 1].axis("off")
    
    axs[1, 0].imshow(segmented_black_bg)
    axs[1, 0].set_title("Segmentasi Objek")
    axs[1, 0].axis("off")
    
    axs[1, 2].imshow(image_rgba)
    axs[1, 2].set_title("Remove BG")
    axs[1, 2].axis("off")
    
    plt.tight_layout()
    plt.show()

# Path gambar
image_path = "tomat.jpg"

# Proses gambar
process_image(image_path)
