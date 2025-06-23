import cv2
import matplotlib.pyplot as plt
import numpy as np

# Membaca gambar grayscale
grayscale_image = cv2.imread('program/tomat/matang/1.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan colormap (cth: 'jet') untuk pewarnaan
colorized_image = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)

# Tampilkan gambar grayscale
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Tampilkan gambar dengan colormap (berwarna)
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))  # Ubah BGR ke RGB agar sesuai dengan Matplotlib
plt.title('Colorized Image (RGB)')
plt.axis('off')

plt.show()
