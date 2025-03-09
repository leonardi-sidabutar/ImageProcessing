import numpy as np

def rgb_to_hsi(R, G, B):
    # Normalisasi RGB ke [0,1]
    R, G, B = R / 255.0, G / 255.0, B / 255.0
    print(f"R:{R:.4f}")
    print(f"G:{G:.4f}")
    print(f"B:{B:.4f}")
    print(f"hasil : {(R+B+G)}")
    
    # Hitung Intensitas (I)
    I = (R + G + B) / 3
    
    # Hitung Saturasi (S)
    min_RGB = min(R, G, B)
    if R + G + B == 0:
        S = 0
    else:
        S = 1 - (3 * min_RGB) / (R + G + B)
        print("min",(3 * min_RGB) / (R + G + B))
    
    # Hitung Hue (H)
    num = 0.5 * ((R - G) + (R - B))
    denom = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6  # Tambah epsilon untuk hindari pembagian nol
    theta = np.arccos(num / denom)
    
    if B > G:
        H = 360 - np.degrees(theta)
    else:
        H = np.degrees(theta)
    
    print(f"cos -1:{np.arccos(0.9981)}")
    print(f"H:{np.degrees(np.arccos(0.9981))}")
    
    # Normalisasi Hue ke [0,1]
    H /= 360.0
    
    return H, S, I

# Input RGB
R, G, B = 94, 4, 3
H, S, I = rgb_to_hsi(R, G, B)

# Cetak hasil
print(f"RGB ({R}, {G}, {B}) ke HSI:")

print(f"Hue: {H:.4f} (dalam rentang 0-1)")
print(f"Saturation: {S:.4f}")
print(f"Intensity: {I:.4f}")
