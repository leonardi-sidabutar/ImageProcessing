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

<<<<<<< HEAD
<<<<<<< HEAD
    print(f"hasil :{num/denom}")
    print(f"theta :{theta}")
=======
    print(num)
    print(denom)
    print(np.degrees(np.arccos(0.9999975480829351)))
>>>>>>> 6542af3a580728661cff4b9e1046de50f3d7a970
    
=======
    print(f"theta = :{360 - (theta * (180/3.14))}")
   
>>>>>>> b6fca853c94ed2dcbf8b86643a38ff7df6f8a402
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
<<<<<<< HEAD
R, G, B = 118, 4, 3
=======
R, G, B = 133, 0, 1
>>>>>>> b6fca853c94ed2dcbf8b86643a38ff7df6f8a402
H, S, I, Hdeg = rgb_to_hsi(R, G, B)

# Cetak hasil
print("===========")
print(f"H = {Hdeg:.4f} (dalam derajat)")
print(f"H = {H:.4f} (dalam rentang 0-1)")
print(f"S = {S:.4f}")
print(f"I = {I:.4f}")