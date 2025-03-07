# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog, Label, Button, Frame
# from PIL import Image, ImageTk
# import matplotlib.pyplot as plt

# def open_image():
#     global img_path, img_rgb
#     img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
#     if img_path:
#         img = Image.open(img_path)
#         img = img.resize((150, 150))  # Resize for display
#         img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#         img_tk = ImageTk.PhotoImage(img)
#         lbl_original.configure(image=img_tk)
#         lbl_original.image = img_tk

# def rgb_to_hsi(image):
#     image = image.astype(np.float32) / 255
#     R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
#     I = (R + G + B) / 3
#     min_RGB = np.minimum(np.minimum(R, G), B)
#     S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB
    
#     num = 0.5 * ((R - G) + (R - B))
#     denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
#     theta = np.arccos(np.clip(num / denom, -1, 1))
#     H = np.where(B > G, 2 * np.pi - theta, theta) / (2 * np.pi)  # Normalize to [0,1]
    
#     return np.stack([H, S, I], axis=-1)

# def process_image():
#     global hsi_img
#     if img_rgb is not None:
#         hsi_img = rgb_to_hsi(img_rgb)
#         H_mean, S_mean, I_mean = np.mean(hsi_img[:, :, 0]), np.mean(hsi_img[:, :, 1]), np.mean(hsi_img[:, :, 2])
#         entry_h.delete(0, tk.END)
#         entry_s.delete(0, tk.END)
#         entry_i.delete(0, tk.END)
#         entry_h.insert(0, f"{H_mean:.4f}")
#         entry_s.insert(0, f"{S_mean:.4f}")
#         entry_i.insert(0, f"{I_mean:.4f}")
        
#         plt.imsave("hsi_output.jpg", hsi_img)
#         img_hsi = Image.open("hsi_output.jpg")
#         img_hsi = img_hsi.resize((150, 150))
#         img_hsi_tk = ImageTk.PhotoImage(img_hsi)
#         lbl_hsi.configure(image=img_hsi_tk)
#         lbl_hsi.image = img_hsi_tk

# def reset():
#     lbl_original.configure(image='')
#     lbl_original.image = None
#     lbl_hsi.configure(image='')
#     lbl_hsi.image = None
#     entry_h.delete(0, tk.END)
#     entry_s.delete(0, tk.END)
#     entry_i.delete(0, tk.END)

# root = tk.Tk()
# root.title("Segmentasi Tingkat Kematangan Buah Tomat")

# frame_left = Frame(root)
# frame_left.grid(row=0, column=0, padx=10, pady=10)
# Button(frame_left, text="Pilih Gambar", command=open_image).pack(pady=5)
# Button(frame_left, text="Proses", command=process_image).pack(pady=5)
# Button(frame_left, text="Reset", command=reset).pack(pady=5)

# frame_right = Frame(root)
# frame_right.grid(row=0, column=1, padx=10, pady=10)
# Label(frame_right, text="Gambar Asli").grid(row=0, column=0)
# Label(frame_right, text="Hasil Konversi HSI").grid(row=0, column=1)
# lbl_original = Label(frame_right)
# lbl_original.grid(row=1, column=0, padx=5, pady=5)
# lbl_hsi = Label(frame_right)
# lbl_hsi.grid(row=1, column=1, padx=5, pady=5)

# frame_bottom = Frame(root)
# frame_bottom.grid(row=1, column=0, columnspan=2, pady=10)
# Label(frame_bottom, text="H").grid(row=0, column=0)
# Label(frame_bottom, text="S").grid(row=0, column=1)
# Label(frame_bottom, text="I").grid(row=0, column=2)
# entry_h = tk.Entry(frame_bottom, width=10)
# entry_h.grid(row=1, column=0)
# entry_s = tk.Entry(frame_bottom, width=10)
# entry_s.grid(row=1, column=1)
# entry_i = tk.Entry(frame_bottom, width=10)
# entry_i.grid(row=1, column=2)

# root.mainloop()


# HSI di pisah ==================================================================================================================================================

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os

def open_image():
    global img_path, img_rgb
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if img_path:
        img = Image.open(img_path)
        img = img.resize((150, 150))  # Resize for display
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(img)
        lbl_original.configure(image=img_tk)
        lbl_original.image = img_tk

def rgb_to_hsi(image):
    image = image.astype(np.float32) / 255
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    I = (R + G + B) / 3
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB
    
    num = 0.5 * ((R - G) + (R - B))
    denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / denom, -1, 1))
    H = np.where(B > G, 2 * np.pi - theta, theta) / (2 * np.pi)  # Normalize to [0,1]
    
    return np.stack([H, S, I], axis=-1)

def process_image():
    global hsi_img
    if img_rgb is not None:
        hsi_img = rgb_to_hsi(img_rgb)
        H, S, I = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]
        H_mean, S_mean, I_mean = np.mean(H), np.mean(S), np.mean(I)
        entry_h.delete(0, tk.END)
        entry_s.delete(0, tk.END)
        entry_i.delete(0, tk.END)
        entry_h.insert(0, f"{H_mean:.4f}")
        entry_s.insert(0, f"{S_mean:.4f}")
        entry_i.insert(0, f"{I_mean:.4f}")

        # Memubat Folder Hasil
        output_folder = "hasil"
        os.makedirs(output_folder,exist_ok=True)
        
        # Simpan Hasil Proses Konversi
        plt.imsave("hasil/h_output.jpg", H, cmap="hsv")
        plt.imsave("hasil/s_output.jpg", S, cmap="gray")
        plt.imsave("hasil/i_output.jpg", I, cmap="gray")
        
        img_h = Image.open(os.path.join(output_folder,"h_output.jpg")).resize((150, 150))
        img_s = Image.open(os.path.join(output_folder,"s_output.jpg")).resize((150, 150))
        img_i = Image.open(os.path.join(output_folder,"i_output.jpg")).resize((150, 150))
        
        img_h_tk = ImageTk.PhotoImage(img_h)
        img_s_tk = ImageTk.PhotoImage(img_s)
        img_i_tk = ImageTk.PhotoImage(img_i)
        
        lbl_h.configure(image=img_h_tk)
        lbl_h.image = img_h_tk
        lbl_s.configure(image=img_s_tk)
        lbl_s.image = img_s_tk
        lbl_i.configure(image=img_i_tk)
        lbl_i.image = img_i_tk

def reset():
    lbl_original.configure(image='')
    lbl_original.image = None
    lbl_h.configure(image='')
    lbl_h.image = None
    lbl_s.configure(image='')
    lbl_s.image = None
    lbl_i.configure(image='')
    lbl_i.image = None
    entry_h.delete(0, tk.END)
    entry_s.delete(0, tk.END)
    entry_i.delete(0, tk.END)

root = tk.Tk()
root.title("Segmentasi Tingkat Kematangan Buah Tomat")

frame_left = Frame(root)
frame_left.grid(row=0, column=0, padx=10, pady=10)
Button(frame_left, text="Pilih Gambar", command=open_image).pack(pady=5)
Button(frame_left, text="Proses", command=process_image).pack(pady=5)
Button(frame_left, text="Reset", command=reset).pack(pady=5)

frame_right = Frame(root)
frame_right.grid(row=0, column=1, padx=10, pady=10)
Label(frame_right, text="Gambar Asli").grid(row=0, column=0)
Label(frame_right, text="Hasil Konversi HSI").grid(row=0, column=1)
lbl_original = Label(frame_right)
lbl_original.grid(row=1, column=0, padx=5, pady=5)
lbl_hsi = Label(frame_right)
lbl_hsi.grid(row=1, column=1, padx=5, pady=5)

frame_hsi = Frame(root)
frame_hsi.grid(row=2, column=0, columnspan=2, pady=10)
Label(frame_hsi, text="H").grid(row=0, column=0)
Label(frame_hsi, text="S").grid(row=0, column=1)
Label(frame_hsi, text="I").grid(row=0, column=2)
lbl_h = Label(frame_hsi)
lbl_h.grid(row=1, column=0, padx=5, pady=5)
lbl_s = Label(frame_hsi)
lbl_s.grid(row=1, column=1, padx=5, pady=5)
lbl_i = Label(frame_hsi)
lbl_i.grid(row=1, column=2, padx=5, pady=5)

frame_bottom = Frame(root)
frame_bottom.grid(row=3, column=0, columnspan=2, pady=10)
Label(frame_bottom, text="H").grid(row=0, column=0)
Label(frame_bottom, text="S").grid(row=0, column=1)
Label(frame_bottom, text="I").grid(row=0, column=2)
entry_h = tk.Entry(frame_bottom, width=10)
entry_h.grid(row=1, column=0)
entry_s = tk.Entry(frame_bottom, width=10)
entry_s.grid(row=1, column=1)
entry_i = tk.Entry(frame_bottom, width=10)
entry_i.grid(row=1, column=2)

root.mainloop()
