import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # Untuk menampilkan gambar

def pilih_gambar():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))  # Ubah ukuran gambar
        img = ImageTk.PhotoImage(img)
        label_gambar.config(image=img)
        label_gambar.image = img  # Simpan referensi gambar

def submit_data():
    nama = entry_nama.get()
    umur = entry_umur.get()
    messagebox.showinfo("Data Tersimpan", f"Nama: {nama}\nUmur: {umur}")

# Buat Window
root = tk.Tk()
root.title("Form Input dengan Gambar")
root.geometry("800x800")

# Label & Entry Nama
tk.Label(root, text="Nama:").pack()
entry_nama = tk.Entry(root)
entry_nama.pack()

# Label & Entry Umur
tk.Label(root, text="Umur:").pack()
entry_umur = tk.Entry(root)
entry_umur.pack()

# Tombol Pilih Gambar
btn_gambar = tk.Button(root, text="Pilih Gambar", command=pilih_gambar)
btn_gambar.pack()

# Label untuk Menampilkan Gambar
label_gambar = tk.Label(root)
label_gambar.pack()

# Tombol Submit
btn_submit = tk.Button(root, text="Submit", command=submit_data)
btn_submit.pack()

# Jalankan Aplikasi
root.mainloop()
