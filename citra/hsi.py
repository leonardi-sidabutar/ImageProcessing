# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Baca citra RGB
# rgb_img = cv2.imread("tomat.jpg")
# rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Ubah dari BGR ke RGB
# rgb_img = rgb_img.astype(np.float32) / 255  # Normalisasi ke [0,1]

# # Pisahkan saluran warna
# R, G, B = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]

# # Hitung Intensity (I)
# I = (R + G + B) / 3

# # Hitung Saturation (S)
# min_RGB = np.minimum(np.minimum(R, G), B)
# S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB  # Tambahkan epsilon kecil untuk mencegah divisi nol

# # Hitung Hue (H)
# num = 0.5 * ((R - G) + (R - B))
# denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # Hindari pembagian dengan nol
# theta = np.arccos(num / denom)

# H = np.degrees(theta)  # Konversi ke derajat
# H[B > G] = 360 - H[B > G]  # Koreksi Hue jika B > G
# H = H / 360  # Normalisasi ke [0,1]

# # Gabungkan menjadi citra HSI
# hsi_img = np.stack([H, S, I], axis=-1)

# # Tampilkan hasil
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0, 0].imshow(rgb_img)
# ax[0, 0].set_title("Citra RGB")
# ax[0, 1].imshow(H, cmap="hsv")
# ax[0, 1].set_title("Hue")
# ax[1, 0].imshow(S, cmap="gray")
# ax[1, 0].set_title("Saturation")
# ax[1, 1].imshow(I, cmap="gray")
# ax[1, 1].set_title("Intensity")
# plt.show()


# Revisi 1

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca citra RGB
rgb_img = cv2.imread("tomat.jpg")
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Ubah dari BGR ke RGB
rgb_img = rgb_img.astype(np.float32) / 255  # Normalisasi ke [0,1]

# Pisahkan saluran warna
R, G, B = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]

# Hitung Intensity (I)
I = (R + G + B) / 3

# Hitung Saturation (S)
min_RGB = np.minimum(np.minimum(R, G), B)
S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB  # Tambahkan epsilon kecil untuk mencegah divisi nol

# Hitung Hue (H) dalam radian
num = 0.5 * ((R - G) + (R - B))
denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # Hindari pembagian dengan nol
theta = np.arccos(np.clip(num / denom, -1, 1))  # Gunakan np.clip untuk menghindari error

H = np.where(B > G, 2 * np.pi - theta, theta)  # Hue tetap dalam radian
H = H / (2 * np.pi)  # Normalisasi ke [0,1]

# Gabungkan menjadi citra HSI
hsi_img = np.stack([H, S, I], axis=-1)

# Tampilkan hasil
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(rgb_img)
ax[0, 0].set_title("Citra RGB")
ax[0, 1].imshow(H, cmap="hsv")
ax[0, 1].set_title("Hue (Radian, Normalized)")
ax[1, 0].imshow(S, cmap="gray")
ax[1, 0].set_title("Saturation")
ax[1, 1].imshow(I, cmap="gray")
ax[1, 1].set_title("Intensity")
plt.show()