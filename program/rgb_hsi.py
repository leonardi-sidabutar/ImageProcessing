import numpy as np

def rgb_to_hsi(R, G, B):
    # Normalisasi RGB ke [0,1]
    R, G, B = R / 255.0, G / 255.0, B / 255.0
    print(f"R' = {R:.4f}")
    print(f"G' = {G:.4f}")
    print(f"B' = {B:.4f}")
     
    # Hitung Intensitas (I)
    I = (R + G + B) / 3
    
    # Hitung Saturasi (S)
    min_RGB = min(R, G, B)
    if R + G + B == 0:
        S = 0
    else:
        S = 1 - (3 * min_RGB) / (R + G + B)
    
    # Hitung Hue (H)
    num = 0.5 * ((R - G) + (R - B))
    denom = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6  # Tambah epsilon untuk hindari pembagian nol
    theta = np.arccos(num / denom)

    a = S
   
    print(f"TESST ======================== {a:.4f}") 

    if B > G:
        H = 360 - np.degrees(theta)
        print('B > G')
    else:
        H = np.degrees(theta)
        print('B < G')
    
   
    Hdeg = H


    # Normalisasi Hue ke [0,1]
    H /= 360.0

   
    return H, S, I, Hdeg

# Input RGB
R, G, B = 91, 47, 2

H, S, I, Hdeg = rgb_to_hsi(R, G, B)

# Cetak hasil
print("===========")
print(f"H = {Hdeg:.4f} (dalam derajat)")
print(f"H = {H:.4f} (dalam rentang 0-1)")
print(f"S = {S:.4f}")
print(f"I = {I:.4f}")