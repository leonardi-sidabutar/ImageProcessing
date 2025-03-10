import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. BACA CITRA ===
img = cv2.imread("tomat/setengah/6.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi ke RGB untuk ditampilkan di matplotlib

# === 2. KONVERSI KE GRAYSCALE ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === 3. THRESHOLDING (OBJEK PUTIH, BACKGROUND HITAM) ===
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# === 4. PERBAIKI MASK DENGAN MORPHOLOGICAL OPERATION ===
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# === 5. SEGMENTASI (HAPUS BACKGROUND) ===
segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# === 6. TAMPILKAN HASIL ===
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img_rgb)
ax[0].set_title("Citra Asli")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask Biner")

ax[2].imshow(segmented)
ax[2].set_title("Citra Hasil Segmentasi")

plt.show()
