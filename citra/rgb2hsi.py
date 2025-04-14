import cv2
import numpy as np

def rgb_to_hsi(image):
    image = image.astype(np.float32) / 255.0  # Normalisasi ke [0,1]
    R, G, B = cv2.split(image)

    # Hitung Intensitas (I)
    I = (R + G + B) / 3.0

    # Hitung Saturasi (S)
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB  # Hindari pembagian dengan nol

    # Hitung Hue (H)
    num = 0.5 * ((R - G) + (R - B))
    denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # Hindari pembagian nol
    theta = np.arccos(num / denom)  # Hasil dalam radian

    H = np.degrees(theta)  # Ubah radian ke derajat
    H[B > G] = 360 - H[B > G]  # Koreksi hue jika B > G
    H = H / 360  # Normalisasi ke rentang [0,1]

    # Hitung nilai rata-rata dari seluruh gambar
    avg_H = np.mean(H)
    avg_S = np.mean(S)
    avg_I = np.mean(I)

    return H, S, I, avg_H, avg_S, avg_I

# Baca gambar
image = cv2.imread("tomat/matang/1.jpg")  # Ubah dengan path gambar Anda
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ubah dari BGR ke RGB

# Konversi ke HSI dan hitung rata-rata
H, S, I, avg_H, avg_S, avg_I = rgb_to_hsi(image)

print(f"Rata-rata H: {avg_H:.4f}")
print(f"Rata-rata S: {avg_S:.4f}")
print(f"Rata-rata I: {avg_I:.4f}")
