import cv2
import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk  # Tambahkan PIL untuk memanipulasi gambar

class TomatoSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pengolahan Citra Kematangan Tomat Metode HSI")
        root.geometry("695x560")

    # Frame Title --------------------------------------------------------------------------------------------------------
        self.frame_title = tk.Frame(root, padx=10, pady=10)
        self.frame_title.grid(row=0, column=0, columnspan=4, sticky="ew")
        tk.Label(
            self.frame_title,
            text="Pengolahan Citra Menentukan Tingkat Kematangan Buah Tomat dengan Metode HSI",
            font=("Arial", 12, "bold")
        ).pack(pady=2)
        
    # Frame Pengolahan --------------------------------------------------------------------------------------------------------
        self.frame_processing = tk.Frame(root, padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
        self.frame_processing.grid(row=1, column=0, rowspan=4, sticky="ns")

        # Memuat gambar menggunakan Pillow
        image = Image.open("tgd.png")  # Pastikan file logo berada dalam folder yang benar
        image = image.resize((75,75))
        self.logo = ImageTk.PhotoImage(image)  # Simpan gambar sebagai atribut instance

        # Menampilkan gambar di Label (dalam frame_logo)
        logo_label = tk.Label(self.frame_processing, image=self.logo)
        logo_label.pack(pady=10)
        
        # Simpan gambar RGB (placeholder untuk kebutuhan lainnya)
        self.image_rgb = None
       
        tk.Button(self.frame_processing, text="Pilih Gambar", command=self.load_image).pack(pady=2)
        self.label_filename = tk.Label(self.frame_processing, text="[Nama Gambar]", relief=tk.SUNKEN, width=20)
        self.label_filename.pack(pady=5)
        tk.Button(self.frame_processing, text="Segmentasi", width=15, command=self.segmentasi).pack(pady=2)
        tk.Button(self.frame_processing, text="Konversi HSI", width=15).pack(pady=2)
        tk.Button(self.frame_processing, text="Proses", width=15, command=self.process).pack(pady=2)
        tk.Button(self.frame_processing, text="Reset", width=15, command=self.reset).pack(pady=2)
        
    # Frame Gambar --------------------------------------------------------------------------------------------------------
        self.frame_images = tk.Frame(root, padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
        self.frame_images.grid(row=1, column=1, columnspan=3)

        # Label kecil di atas setiap Canvas
        labels = ["Gambar RGB", "Hasil Segmentasi", "Hasil Konversi HSI"]
        for col, text in enumerate(labels):
            tk.Label(self.frame_images, text=text, font=("Arial", 10)).grid(row=0, column=col, pady=(0, 2))

        # Canvas utama (RGB, Segmentasi, HSI)
        self.canvas_rgb = tk.Canvas(self.frame_images, width=150, height=150, bg="#333", border=1)
        self.canvas_rgb.grid(row=1, column=0, padx=5, pady=5)

        self.canvas_segmentasi = tk.Canvas(self.frame_images, width=150, height=150, bg="#333", border=1)
        self.canvas_segmentasi.grid(row=1, column=1, padx=5, pady=5)

        self.canvas_hsi = tk.Canvas(self.frame_images, width=150, height=150, bg="#333", border=1)
        self.canvas_hsi.grid(row=1, column=2, padx=5, pady=5)

        # Label kecil di atas setiap Canvas komponen HSI
        hsi_labels = ["H (Hue)", "S (Saturation)", "I (Intensity)"]
        for col, text in enumerate(hsi_labels):
            tk.Label(self.frame_images, text=text, font=("Arial", 10)).grid(row=2, column=col, pady=(0, 2))

        # Canvas komponen HSI
        self.canvas_hue = tk.Canvas(self.frame_images, width=150, height=150, bg="#ddd", border=1)
        self.canvas_hue.grid(row=3, column=0, padx=5, pady=5)

        self.canvas_saturation = tk.Canvas(self.frame_images, width=150, height=150, bg="#ddd", border=1)
        self.canvas_saturation.grid(row=3, column=1, padx=5, pady=5)

        self.canvas_intensity = tk.Canvas(self.frame_images, width=150, height=150, bg="#ddd", border=1)
        self.canvas_intensity.grid(row=3, column=2, padx=5, pady=5)
        
    # Frame Logo ---------------------------------------------------------------------------------------------------------

        # self.frame_logo = tk.Frame(root,padx=5, pady=5, relief=tk.RIDGE, borderwidth=2)
        # self.frame_logo.grid(row=4, column=0, sticky="ew")
        # # Memuat gambar menggunakan Pillow
        # image = Image.open("tgd.png")  # Pastikan file logo berada dalam folder yang benar
        # image = image.resize((75,75))
        # self.logo = ImageTk.PhotoImage(image)  # Simpan gambar sebagai atribut instance

        # # Menampilkan gambar di Label (dalam frame_logo)
        # logo_label = tk.Label(self.frame_logo, image=self.logo)
        # logo_label.pack(pady=5)

    # Frame Hasil --------------------------------------------------------------------------------------------------------
        self.frame_results = tk.Frame(root, padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
        self.frame_results.grid(row=4, column=1, columnspan=3, sticky="ew")

        # Frame dalam untuk meratakan ke tengah
        self.inner_frame = tk.Frame(self.frame_results)
        self.inner_frame.pack(expand=True)

        # Baris pertama (H, S, I)
        tk.Label(self.inner_frame, text="H").grid(row=0, column=0, padx=5, pady=2)
        tk.Entry(self.inner_frame, width=5).grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.inner_frame, text="S").grid(row=0, column=2, padx=5, pady=2)
        tk.Entry(self.inner_frame, width=5).grid(row=0, column=3, padx=5, pady=2)
        tk.Label(self.inner_frame, text="I").grid(row=0, column=4, padx=5, pady=2)
        tk.Entry(self.inner_frame, width=5).grid(row=0, column=5, padx=5, pady=2)

        # Baris kedua (Tingkat Kematangan)
        tk.Label(self.inner_frame, text="Tingkat Kematangan").grid(row=1, column=0, columnspan=6, pady=(10, 2))
        self.entry_kematangan = tk.Entry(self.inner_frame, width=25, justify="center", state="readonly")
        self.entry_kematangan.grid(row=2, column=0, columnspan=6, pady=2)

    def load_image(self):
        """Memuat gambar yang dipilih dan menampilkannya di canvas_rgb"""
        self.filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.filename:
            self.label_filename.config(text=self.filename.split("/")[-1])  # Update label nama gambar

            # Tampilkan loading
            self.label_loading = tk.Label(self.frame_processing, text="Loading...", font=("Arial", 10))
            self.label_loading.pack()
            self.root.update_idletasks()
            
            # Load gambar menggunakan PIL
            image = Image.open(self.filename)
            image = image.resize((150,150))

            # Dapatkan ukuran canvas
            canvas_width = self.canvas_rgb.winfo_width()
            canvas_height = self.canvas_rgb.winfo_height()

            # Resize gambar agar sesuai dengan canvas (thumbnail mempertahankan rasio aspek)
            image.thumbnail((canvas_width, canvas_height))  

            # Konversi ke format Tkinter
            self.image_rgb = ImageTk.PhotoImage(image)

            # Hapus gambar lama sebelum menggambar yang baru
            self.canvas_rgb.delete("all")  

            # Tampilkan gambar di tengah canvas
            self.canvas_rgb.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.image_rgb)


            # Sembunyikan loading
            self.label_loading.pack_forget()

    def segmentasi(self):
        if self.filename:
            # === 1. Baca Citra ===
            img = cv2.imread(self.filename)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # === 2. KONVERSI KE GRAYSCALE ===
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # === 3. THRESHOLDING (OBJEK PUTIH, BACKGROUND HITAM) ===
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # === 4. PERBAIKI MASK DENGAN MORPHOLOGICAL OPERATION ===
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # === 5. SEGMENTASI (HAPUS BACKGROUND) ===
            segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

            # Konversi ke format PIL
            segmented_pil = Image.fromarray(segmented_rgb)
            segmented_pil = segmented_pil.resize((150, 150))

            # Konversi ke format Tkinter
            self.segmented_img = ImageTk.PhotoImage(segmented_pil)

            # Bersihkan canvas dan tampilkan gambar
            self.canvas_segmentasi.delete("all")
            self.canvas_segmentasi.create_image(75, 75, anchor=tk.CENTER, image=self.segmented_img)

    # Mencoba function baru
    def process(self):
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.insert(0,"Matang")
        self.entry_kematangan.config(state="readonly")

    def reset(self):
        """Reset tampilan ke kondisi awal"""
        self.label_filename.config(text="[Nama Gambar]")
        self.canvas_rgb.delete("all")  # Hapus gambar dari canvas
        
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.config(state="readonly")

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoSegmentationApp(root)
    root.mainloop()
