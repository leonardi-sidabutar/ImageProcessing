import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. BACA CITRA ===
img = cv2.imread("tomat/matang/1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# === 2. SEGMENTASI OBJEK TOMAT (MENGGUNAKAN HSV) ===
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

# === 3. KONVERSI SEGMENTASI KE HSI ===
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

# Konversi hasil segmentasi ke HSI
hsi_img = rgb_to_hsi(segmented_rgb)
H, S, I = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]

# === 4. HITUNG NILAI RATA-RATA HUE UNTUK OBJEK TOMAT ===
H_tomato = H[mask > 0]  # Ambil hanya piksel yang masuk dalam mask
# average_Hue = np.mean(H_tomato) * 360  # Konversi ke derajat
average_Hue = np.mean(H_tomato)   # Konversi ke rentang nilai [0-1]
print(f"Rata-rata nilai Hue untuk objek tomat: {average_Hue:.2f} derajat")

# === 5. PILIH 8 PIXEL ACAK DAN TAMPILKAN NILAINYA ===
pixel_indices = np.column_stack(np.where(mask > 0))  # Ambil indeks piksel dalam mask
selected_pixels = pixel_indices[np.random.choice(len(pixel_indices), 8, replace=False)]

print("8 Pixel Acak dan Nilai Hue-nya:")
for idx, (y, x) in enumerate(selected_pixels):
    hue_value = H[y, x] # Konversi ke derajat
    rgb_value = segmented_rgb[y, x]  # Ambil nilai RGB
    print(f"Pixel {idx+1}: Posisi ({x}, {y}), RGB = {rgb_value}, Hue = {hue_value:.2f}")

# Tandai pixel pada gambar Hue
fig, ax = plt.subplots()
ax.imshow(H, cmap="hsv")
ax.set_title("Komponen Hue dengan Titik Acak")
for (y, x) in selected_pixels:
    ax.scatter(x, y, color='red', s=50, edgecolors='black')
plt.show()

# === 6. TAMPILKAN HASIL ===
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].imshow(img_rgb)
ax[0, 0].set_title("Citra RGB Asli")
ax[0, 1].imshow(mask, cmap="gray")
ax[0, 1].set_title("Mask Segmentasi")
ax[0, 2].imshow(segmented_rgb)
ax[0, 2].set_title("Citra Hasil Segmentasi")
ax[1, 0].imshow(H, cmap="hsv")
ax[1, 0].set_title("Komponen Hue (H)")
ax[1, 1].imshow(S, cmap="gray")
ax[1, 1].set_title("Komponen Saturation (S)")
ax[1, 2].imshow(I, cmap="gray")
ax[1, 2].set_title("Komponen Intensity (I)")
plt.show()
