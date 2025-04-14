import cv2

# Baca citra RGB
image_rgb = cv2.imread('tomat/matang/1.jpg')  # Ganti 'gambar.jpg' dengan nama file citra Anda

# Ubah citra RGB menjadi grayscale
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

# Simpan hasil citra grayscale
cv2.imwrite('gambar_grayscale.jpg', image_gray)

print("Citra berhasil diubah menjadi grayscale dan disimpan sebagai 'gambar_grayscale.jpg'.")
