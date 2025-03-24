import numpy as np

theta_rad = np.arccos(0.99997)  # 1.047 radian
theta_deg = np.degrees(theta_rad)  # 60 derajat

print(f"nilai theta :{theta_rad}")
print(theta_deg/360)  # Output: 60.0