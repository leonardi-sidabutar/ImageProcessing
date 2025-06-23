import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar dan konversi ke HSV dan RGB
img = cv2.imread("tomat/matang/1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Ambil ukuran gambar dan tentukan ROI 20x20 pixel di tengah
h, w = hsv.shape[:2]
cx, cy = w // 2, h // 2
roi_size = 20
x1, x2 = cx - roi_size // 2, cx + roi_size // 2
y1, y2 = cy - roi_size // 2, cy + roi_size // 2

# Ambil ROI dan hitung rata-rata HSV
roi = hsv[y1:y2, x1:x2]
mean_h = int(np.mean(roi[:, :, 0]))
mean_s = int(np.mean(roi[:, :, 1]))
mean_v = int(np.mean(roi[:, :, 2]))

# Cetak rata-rata HSV
print(f"Rata-rata HSV dari ROI 20x20: H={mean_h}, S={mean_s}, V={mean_v}")

# Buat rentang HSV adaptif
delta_h, delta_s, delta_v = 10, 50, 50
lower = np.array([max(mean_h - delta_h, 0), max(mean_s - delta_s, 0), max(mean_v - delta_v, 0)])
upper = np.array([min(mean_h + delta_h, 179), min(mean_s + delta_s, 255), min(mean_v + delta_v, 255)])

# Buat mask dan bersihkan
mask = cv2.inRange(hsv, lower, upper)
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

# Segmentasi RGB berdasarkan mask
segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)

# Gambar kotak pada gambar asli
img_roi_box = img_rgb.copy()
cv2.rectangle(img_roi_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Crop ROI RGB untuk ditampilkan
roi_rgb = img_rgb[y1:y2, x1:x2]

# Tampilkan hasil
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(img_roi_box)
plt.title("Gambar Asli + Kotak 20x20")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(roi_rgb)
plt.title(f"ROI 20x20 (H={mean_h}, S={mean_s}, V={mean_v})")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask_clean, cmap="gray")
plt.title("Mask dari HSV Area Tengah")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(segmented)
plt.title("Hasil Segmentasi")
plt.axis("off")

plt.tight_layout()
plt.show()
