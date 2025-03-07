import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. BACA CITRA ===
img = cv2.imread("tomat.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# === 2. SEGMENTASI OBJEK TOMAT ===
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Bersihkan noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Terapkan mask ke citra asli
segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# === 3. KONVERSI KE HSI ===
def rgb_to_hsi(image):
    image = image.astype(np.float32) / 255.0
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    I = (R + G + B) / 3
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB  
    num = 0.5 * ((R - G) + (R - B))
    denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(num / denom)
    H = np.degrees(theta)
    H[B > G] = 360 - H[B > G]
    H = H / 360  
    return np.stack([H, S, I], axis=-1)

hsi_img = rgb_to_hsi(segmented_rgb)
H, S, I = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]

# === 4. PILIH 8 PIXEL ACAK ===
pixel_indices = np.column_stack(np.where(mask > 0))  # Ambil indeks piksel dalam mask
selected_pixels = pixel_indices[np.random.choice(len(pixel_indices), 8, replace=False)]

# === 5. TAMPILKAN PIXEL DI CITRA RGB DAN HSI ===
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(segmented_rgb)
ax[0].set_title("Citra RGB")
ax[1].imshow(H, cmap="hsv")
ax[1].set_title("Komponen Hue")
ax[2].imshow(S, cmap="gray")
ax[2].set_title("Komponen Saturation")
ax[3].imshow(I, cmap="gray")
ax[3].set_title("Komponen Intensity")

for (y, x) in selected_pixels:
    for a in ax:
        a.scatter(x, y, facecolors='none', s=50, edgecolors='green')

for idx, (y, x) in enumerate(selected_pixels):
    hue_value = H[y, x] * 360  # Konversi ke derajat
    print(f"Pixel {idx+1}: Posisi ({x}, {y}), Hue = {hue_value:.2f} derajat")

plt.show()
