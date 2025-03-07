import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca citra
img = cv2.imread("tomat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Rentang warna merah/oranye untuk tomat (dapat disesuaikan)
lower_red1 = np.array([0, 100, 50])   # Rentang pertama merah
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 50]) # Rentang kedua merah
upper_red2 = np.array([180, 255, 255])

# Buat mask berdasarkan rentang warna
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2  # Gabungkan kedua rentang merah

# Operasi morfologi untuk membersihkan noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Terapkan mask ke citra asli
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# Tampilkan hasil
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img_rgb)
ax[0].set_title("Citra Asli")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask Tomat")
ax[2].imshow(result)
ax[2].set_title("Segmentasi Tomat")
plt.show()
