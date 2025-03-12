import cv2
import tkinter as tk
import numpy as np
from tkinter import filedialog
from tkinter import messagebox  # Import messagebox
from PIL import Image, ImageTk  # Tambahkan PIL untuk memanipulasi gambar
import matplotlib.pyplot as plt

class TomatoSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pengolahan Citra Kematangan Tomat Metode HSI")
        
        app_width = 695
        app_height = 560

        screen_width = self.root.winfo_screenwidth()        # Lebar Window
        screen_height = self.root.winfo_screenheight()      # Tinggi Window

        x = (screen_width // 2) - (app_width // 2)
        y = (screen_height // 2) - (app_height // 2)

        root.geometry(f"{app_width}x{app_height}+{x}+{y}")

        # Kondisi Awal FileName
        self.filename = False


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
       
        # Tombol View
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
        labels = ["Gambar RGB", "Hasil Segmentasi", "Hasil Morfologi"]
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

    def _rgb2hsi(self):
        image = self.segmented_rgb
        image = image.astype(np.float32) / 255.0
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        I = (R + G + B) / 3
        min_RGB = np.minimum(np.minimum(R, G), B)
        S = 1 - (3 / (R + G + B + 1e-6)) * min_RGB  
        num = 0.5 * ((R - G) + (R - B))
        denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
        theta = np.arccos(num / denom)
        H = np.degrees(theta)
        H[B > G] = 360 - H[B > G]
        H = H / 360  
        return np.stack([H, S, I], axis=-1)

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

    # Tombol Segmentasi
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
            self.segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

            # Konversi ke format PIL
            segmented_pil = Image.fromarray(self.segmented_rgb)
            segmented_pil = segmented_pil.resize((150, 150))

            # Konversi ke format Tkinter
            self.segmented_img = ImageTk.PhotoImage(segmented_pil)

            # Bersihkan canvas dan tampilkan gambar
            self.canvas_segmentasi.delete("all")
            self.canvas_segmentasi.create_image(75, 75, anchor=tk.CENTER, image=self.segmented_img)
        else:
            messagebox.showerror("Peringatan","Anda Belum Memilih Gambar")

    # Tombol Proses Function
    def process(self):

        if self.filename:

            # Melakukan Proses Segmentasi Citra
            self.segmentasi()

            # Melakukan Konversi Citra ke HSI
            hsi_img = self._rgb2hsi()
            self.H = hsi_img[:, :, 0]  # Hue (0 - 1)
            self.S = hsi_img[:, :, 1]  # Saturation (0 - 1)
            self.I = hsi_img[:, :, 2]  # Intensity (0 - 1)

            # ====== Konversi H, S, I ke gambar dengan cara yang sama ======
            def array_to_pil(image_array, cmap):
                """Konversi array ke PIL Image menggunakan Matplotlib untuk konsistensi."""
                fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100)
                ax.imshow(image_array, cmap=cmap)
                ax.axis("off")
                
                fig.canvas.draw()
                pil_image = Image.fromarray(np.array(fig.canvas.renderer.buffer_rgba()), mode="RGBA")
                plt.close(fig)
                
                # Konversi ke mode "RGB" untuk kompatibilitas dengan Tkinter
                return pil_image.convert("RGB")

            # Konversi semua komponen menggunakan metode yang sama
            h_pil = array_to_pil(self.H, "hsv")
            s_pil = array_to_pil(self.S, "gray")
            i_pil = array_to_pil(self.I, "gray")

            # Resize ke ukuran canvas (150x150)
            h_pil = h_pil.resize((150, 150), Image.Resampling.NEAREST)
            s_pil = s_pil.resize((150, 150), Image.Resampling.NEAREST)
            i_pil = i_pil.resize((150, 150), Image.Resampling.NEAREST)

            # Konversi ke Tkinter PhotoImage
            self.h_img = ImageTk.PhotoImage(h_pil)
            self.s_img = ImageTk.PhotoImage(s_pil)
            self.i_img = ImageTk.PhotoImage(i_pil)

            # Bersihkan Canvas sebelum menggambar ulang
            self.canvas_hue.delete("all")
            self.canvas_saturation.delete("all")
            self.canvas_intensity.delete("all")

            # Tampilkan gambar di Canvas
            self.canvas_hue.create_image(75, 75, anchor=tk.CENTER, image=self.h_img)
            self.canvas_saturation.create_image(75, 75, anchor=tk.CENTER, image=self.s_img)
            self.canvas_intensity.create_image(75, 75, anchor=tk.CENTER, image=self.i_img)

            self.entry_kematangan.config(state="normal")
            self.entry_kematangan.delete(0,tk.END)
            self.entry_kematangan.insert(0,"Matang")
            self.entry_kematangan.config(state="readonly")
        else:
            messagebox.showerror("Peringatan","Anda Belum Memilih Gambar")

    # Tombol Reset Function
    def reset(self):
        """Reset tampilan ke kondisi awal"""
        self.filename = False
        self.label_filename.config(text="[Nama Gambar]")

        # Hapus Gambar Dari Canvas
        self.canvas_rgb.delete("all")
        self.canvas_segmentasi.delete("all")
        self.canvas_hue.delete("all")
        self.canvas_saturation.delete("all")
        self.canvas_intensity.delete("all")
        
        # Reset Entry
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.config(state="readonly")

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoSegmentationApp(root)
    root.mainloop()
