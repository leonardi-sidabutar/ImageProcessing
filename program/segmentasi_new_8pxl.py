import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar dan konversi ke HSV dan RGB
img = cv2.imread("tomat/mentah/1.jpg")
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
segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)


# === 6. KONVERSI KE HSI ===
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

# === 7. PILIH 8 PIXEL ACAK ===
pixel_indices = np.column_stack(np.where(mask > 0))  # Ambil indeks piksel dalam mask
selected_pixels = pixel_indices[np.random.choice(len(pixel_indices), 8, replace=False)]

# === 7. PILIH 8 PIXEL KONSTAN ===


# Coordinat Matang
# selected_pixels = np.array([
# [262,510],
# [713,719],
# [161,631],
# [295,325],
# [828,453],
# [951,400],
# [674,880],
# [715,420]
# ])


# Coordinat Setengah_Matang
# selected_pixels = np.array([
# [761, 851],
# [376, 618],
# [198, 489],
# [471, 281],
# [983, 589],
# [221, 362],
# [460, 798],
# [769, 482],
# ])


# Coordinat Mentah
selected_pixels = np.array([
[165, 431],
[307, 339],
[426, 446],
[664, 321],
[487, 197],
[354, 186],
[437, 211],
[555, 429],
])




# === 8. TAMPILKAN PIXEL DI CITRA RGB DAN HSI ===
fig, ax = plt.subplots(1, 4, figsize=(20, 10))
ax[0].imshow(segmented_rgb)
ax[0].set_title("Citra RGB")
ax[1].imshow(H, cmap="hsv")
ax[1].set_title("Komponen Hue")
ax[2].imshow(S, cmap="gray")
ax[2].set_title("Komponen Saturation")
ax[3].imshow(I, cmap="gray")
ax[3].set_title("Komponen Intensity")
# ax[4].imshow(img_rgb)
# ax[4].set_title("Citra RGB")
# ax[5].imshow(mask)
# ax[5].set_title("Mask")
# ax[6].imshow(gray)
# ax[6].set_title("Gray")
# ax[6].imshow(gray)
# ax[6].set_title("Gray")

val_h = []
val_s = []
val_i = []

for (y, x) in selected_pixels:
    for a in ax:
        a.scatter(x, y, facecolors='none', s=50, edgecolors='white')

for idx, (y, x) in enumerate(selected_pixels):
    # hue_value = H[y, x] * 360  # Konversi ke derajat
    hue_value = H[y, x] # Konversi ke derajat
    s_value = S[y,x]
    i_value = I[y,x]
    rgb_value = segmented_rgb[y, x]  # Ambil nilai RGB
    print(f"Pixel {idx+1}: Posisi ({x}, {y}), RGB = {rgb_value}, Hue = {hue_value:.4f}, S = {s_value:.4f}, I = {i_value:.4f}")
    val_h.append(hue_value)
    val_s.append(s_value)
    val_i.append(i_value)

# Hitung Rata-rata
avg_h = np.mean(val_h)
avg_s = np.mean(val_s)
avg_i = np.mean(val_i)

# Ambil hanya piksel yang termasuk dalam objek tomat berdasarkan mask
H_values = H[mask > 0]
S_values = S[mask > 0]
I_values = I[mask > 0]


# Hitung rata-rata HSI
H_mean = np.mean(H_values)
S_mean = np.mean(S_values)
I_mean = np.mean(I_values)

print(f"Rata - rata H = {avg_h:.4f}")
print(f"Rata - rata S = {avg_s:.4f}")
print(f"Rata - rata I = {avg_i:.4f}")

print("====================================")

print(f"Rata-rata Hue: {H_mean:.2f}")
print(f"Rata-rata Saturation: {S_mean:.4f}")
print(f"Rata-rata Intensity: {I_mean:.4f}")

plt.show()
