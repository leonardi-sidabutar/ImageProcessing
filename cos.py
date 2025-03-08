import math
theta_rad = math.acos(0.999)  # Hasil dalam radian
theta_deg = math.degrees(theta_rad)  # Konversi ke derajat
theta_res = theta_deg / 360
print(f"Theta dalam radian: {theta_rad}")
print(f"Theta dalam derajat: {theta_deg}")
print(f"Theta dalam 0-1: {theta_res}")