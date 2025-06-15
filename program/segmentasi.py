import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar
image_path = "test/sample_matang2.jpg"  # Ganti sesuai path gambarmu
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# === 1. Segmentasi warna merah (tomat matang) ===
# Perluas batas HSV agar lebih toleran terhadap merah-oranye
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([150, 70, 50])
upper_red2 = np.array([180, 255, 255])

mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

# === 2. Bersihkan noise dengan morphological operation ===
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

# === 3. Temukan kontur dan filter bentuk bulat (circularity) ===
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
tomato_mask = np.zeros_like(mask_clean)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if area > 300 and 0.7 < circularity <= 1.2:
        cv2.drawContours(tomato_mask, [cnt], -1, 255, -1)  # hanya kontur valid

# === 4. Segmentasi gambar RGB ===
segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=tomato_mask)

# === 5. Gambar outline kontur pada hasil segmentasi ===
outlined = segmented_rgb.copy()
final_contours, _ = cv2.findContours(tomato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(outlined, final_contours, -1, (0, 255, 0), 2)  # outline hijau

# === 6. Tampilkan hasil ===
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_rgb)
axs[0].set_title("Gambar Asli")
axs[0].axis("off")

axs[1].imshow(tomato_mask, cmap='gray')
axs[1].set_title("Mask Tomat")
axs[1].axis("off")

axs[2].imshow(outlined)
axs[2].set_title("Hasil Segmentasi + Outline")
axs[2].axis("off")

plt.tight_layout()
plt.show()
