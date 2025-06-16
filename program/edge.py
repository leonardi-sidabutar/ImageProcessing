import cv2
import numpy as np

# 1. Baca gambar
img = cv2.imread('tomat.jpg')  # Ganti dengan path gambar kamu
if img is None:
    print("Gambar tidak ditemukan. Pastikan path-nya benar.")
    exit()

original = img.copy()

# 2. Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Lebih halus dari (7, 7)
edges = cv2.Canny(blur, 30, 100)  # Coba threshold yang lebih sensitif

# Debug: tampilkan edges
cv2.imshow('Edges', edges)

# 3. Deteksi kontur
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Jumlah kontur ditemukan: {len(contours)}")

# 4. Filter berdasarkan bentuk (circularity)
def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0: return 0
    return 4 * np.pi * (area / (perimeter ** 2))

mask = np.zeros_like(gray)

detected_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    circ = circularity(cnt)
    if area > 500 and circ > 0.6:  # Turunkan ambang
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        detected_count += 1

print(f"Jumlah kontur lolos filter: {detected_count}")
cv2.imshow('Mask', mask)

# 5. Masking tomat
masked = cv2.bitwise_and(img, img, mask=mask)

# 6. Konversi ke HSV
hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

# 7. Ambil nilai Hue
hue_values = hsv[:, :, 0][mask == 255]
if len(hue_values) == 0:
    print("Tidak ada objek terdeteksi.")
else:
    avg_hue = np.mean(hue_values)
    print(f'Rata-rata Hue: {avg_hue:.2f}')

    # 8. Klasifikasi berdasarkan Hue
    if avg_hue < 20 or avg_hue > 160:
        kematangan = 'Matang'
    elif 35 < avg_hue < 85:
        kematangan = 'Mentah'
    else:
        kematangan = 'Transisi / Tidak diketahui'

    print(f'Tingkat kematangan: {kematangan}')

    # Tampilkan hasil
    cv2.putText(original, kematangan, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow('Original + Label', original)
    cv2.imshow('Segmented Tomat', masked)

cv2.waitKey(0)
cv2.destroyAllWindows()
