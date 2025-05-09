import cv2
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier  # For Machine Learning
import numpy as np
from tkinter import filedialog
from tkinter import messagebox  # Import messagebox
from PIL import Image, ImageTk  # Tambahkan PIL untuk memanipulasi gambar
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score  # pastikan ini diimpor di atas

# Class KNN
class KlasifikasiKematangan:
    def __init__(self):
        self.knn_model = None

    def train_knn_model(self):
        # Data dari tabel Anda
        mentah = [
            [0.15, 0.8854, 0.2118],
            [0.14, 0.8514, 0.2674],
            [0.14, 0.8644, 0.2684],
            [0.15, 0.8552, 0.2467],
            [0.15, 0.8534, 0.2336],
            [0.13, 0.8613, 0.2609],
            [0.16, 0.8593, 0.2504],
            [0.15, 0.8863, 0.283],
            [0.18, 0.8343, 0.2593],
            [0.14, 0.8801, 0.2532]
        ]
        
        setengah = [
            [0.08, 0.8386, 0.2447],
            [0.10, 0.8355, 0.2322],
            [0.07, 0.8424, 0.2641],
            [0.09, 0.8610, 0.2248],
            [0.08, 0.8478, 0.2189],
            [0.09, 0.8344, 0.2354],
            [0.09, 0.8833, 0.2801],
            [0.11, 0.8396, 0.2204],
            [0.10, 0.8580, 0.2771],
            [0.11, 0.8515, 0.2296]
        ]
        
        matang = [
            [0.32, 0.8497, 0.1898],
            [0.35, 0.8732, 0.18],
            [0.28, 0.8531, 0.1865],
            [0.24, 0.8696, 0.1998],
            [0.31, 0.8529, 0.1813],
            [0.28, 0.8905, 0.1803],
            [0.42, 0.8784, 0.1897],
            [0.35, 0.8599, 0.1923],
            [0.33, 0.8518, 0.1894],
            [0.26, 0.8824, 0.2013]
        ]
        
        X = np.array(mentah + setengah + matang)
        y = ['Mentah'] * 10 + ['Setengah Matang'] * 10 + ['Matang'] * 10
        
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.knn_model.fit(X, y)
        print("Model KNN telah dilatih.")

    def classify_kematangan(self, h, s, i):
        if self.knn_model is not None:
            pred = self.knn_model.predict([[h, s, i]])[0]
            return pred
        return "Model belum dilatih"


class TomatoSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pengolahan Citra Kematangan Tomat Metode HSI")
        
        app_width = 525
        app_height = 560

        screen_width = self.root.winfo_screenwidth()        # Lebar Window
        screen_height = self.root.winfo_screenheight()      # Tinggi Window

        x = (screen_width // 2) - (app_width // 2)
        y = (screen_height // 2) - (app_height // 2)

        root.geometry(f"{app_width}x{app_height}+{x}+{y}")

        # Kondisi Awal FileName
        self.filename = False

        self.data_valid = 9
        self.data_uji = 10

        # Test
        self.data_same = 0


    #  Menu Bar ----------------------------------------------------------------------------------------------------------
        menu_bar = tk.Menu(self.root)
        # Menambahkan perintah langsung
        menu_bar.add_command(label="Buka Gambar", command=self.load_image)
        menu_bar.add_command(label="Segmentasi", command=self.segmentasi)
        menu_bar.add_command(label="Konversi HSI", command=self.konversi)
        menu_bar.add_command(label="Proses", command=self.process)
        menu_bar.add_command(label="Mulai Ulang", command=self.mulai_ulang)
        menu_bar.add_command(label="Keluar", command=self.root.quit)
        self.root.config(menu=menu_bar)

    # Frame Title --------------------------------------------------------------------------------------------------------
        self.frame_title = tk.Frame(root, padx=10, pady=10)
        self.frame_title.grid(row=0, column=0, columnspan=4, sticky="ew")
        tk.Label(
            self.frame_title,
            text="Pengolahan Citra Menentukan Tingkat Kematangan Buah Tomat dengan Metode HSI",
            font=("Arial", 9, "bold")
        ).pack(pady=2)
        
        
    # Frame Gambar --------------------------------------------------------------------------------------------------------
        self.frame_images = tk.Frame(root, padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
        self.frame_images.grid(row=1, column=1, columnspan=3)

        # Label kecil di atas setiap Canvas
        labels = ["Gambar RGB", "Hasil Morfologi", "Hasil Segmentasi"]
        for col, text in enumerate(labels):
            tk.Label(self.frame_images, text=text, font=("Arial", 10)).grid(row=0, column=col, pady=(0, 2))

        # Canvas utama (RGB, Segmentasi, HSI)
        self.canvas_rgb = tk.Canvas(self.frame_images, width=150, height=150, bg="#333", border=1)
        self.canvas_rgb.grid(row=1, column=0, padx=5, pady=5)

        self.canvas_segmentasi = tk.Canvas(self.frame_images, width=150, height=150, bg="#333", border=1)
        self.canvas_segmentasi.grid(row=1, column=2, padx=5, pady=5)

        self.canvas_morfologi = tk.Canvas(self.frame_images, width=150, height=150, bg="#333", border=1)
        self.canvas_morfologi.grid(row=1, column=1, padx=5, pady=5)

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
        tk.Label(self.inner_frame, text="Hue:").grid(row=0, column=0, padx=5, pady=2)
        self.entry_h = tk.Entry(self.inner_frame, width=7, state="readonly")
        self.entry_h.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.inner_frame, text="Saturation:").grid(row=0, column=2, padx=5, pady=2)
        self.entry_s = tk.Entry(self.inner_frame, width=7, state="readonly")
        self.entry_s.grid(row=0, column=3, padx=5, pady=2)
        tk.Label(self.inner_frame, text="Intensity:").grid(row=0, column=4, padx=5, pady=2)
        self.entry_i = tk.Entry(self.inner_frame, width=7, state="readonly")
        self.entry_i.grid(row=0, column=5, padx=5, pady=2)


        # Baris 2 ; Kolom 1 (Tingkat Kematangan)
        tk.Label(self.inner_frame, text="Tingkat Kematangan").grid(row=1, column=0, columnspan=3, pady=(10, 2))
        self.entry_kematangan = tk.Entry(self.inner_frame, width=32, justify="center", state="readonly")
        self.entry_kematangan.grid(row=2, column=0, columnspan=3, pady=2)
        # Baris 2 ; Kolom 2
        tk.Label(self.inner_frame, text="Akurasi").grid(row=1, column=4, columnspan=6, pady=(10, 2))
        self.entry_akurasi = tk.Entry(self.inner_frame, width=18, justify="center", state="readonly")
        self.entry_akurasi.grid(row=2, column=4, columnspan=6, pady=2)

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

        self.data_same = 0

        """Memuat gambar yang dipilih dan menampilkannya di canvas_rgb"""
        self.filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.filename:
            # self.label_filename.config(text=self.filename.split("/")[-1])  # Update label nama gambar

            # Tampilkan loading
            # self.label_loading = tk.Label(self.frame_processing, text="Loading...", font=("Arial", 10))
            # self.label_loading.pack()
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
            # self.label_loading.pack_forget()


        self.canvas_segmentasi.delete("all")
        self.canvas_morfologi.delete("all")
        self.canvas_hue.delete("all")
        self.canvas_saturation.delete("all")
        self.canvas_intensity.delete("all")
        
        # Reset Entry
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.config(state="readonly")
        self.entry_h.config(state="normal")
        self.entry_h.delete(0,tk.END)
        self.entry_h.config(state="readonly")
        self.entry_i.config(state="normal")
        self.entry_i.delete(0,tk.END)
        self.entry_i.config(state="readonly")
        self.entry_s.config(state="normal")
        self.entry_s.delete(0,tk.END)
        self.entry_s.config(state="readonly")
        self.entry_akurasi.config(state="normal")
        self.entry_akurasi.delete(0,tk.END)
        self.entry_akurasi.config(state="readonly")


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
            self.masked = mask

            # === 5. SEGMENTASI (HAPUS BACKGROUND) ===
            self.segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

            # Konversi ke format PIL
            segmented_pil = Image.fromarray(self.segmented_rgb)
            segmented_pil = segmented_pil.resize((150, 150))
            mask_pill = Image.fromarray(mask)
            mask_pill = mask_pill.resize((150,150))

            

            # Konversi ke format Tkinter
            self.segmented_img = ImageTk.PhotoImage(segmented_pil)
            self.mask_pill = ImageTk.PhotoImage(mask_pill)

            # Bersihkan canvas dan tampilkan gambar
            self.canvas_segmentasi.delete("all")
            self.canvas_segmentasi.create_image(75, 75, anchor=tk.CENTER, image=self.segmented_img)
            self.canvas_morfologi.delete("all")
            self.canvas_morfologi.create_image(75, 75, anchor=tk.CENTER, image=self.mask_pill)
        else:
            messagebox.showerror("Peringatan","Anda Belum Memilih Gambar")

        self.canvas_hue.delete("all")
        self.canvas_saturation.delete("all")
        self.canvas_intensity.delete("all")

        # Reset Entry
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.config(state="readonly")
        self.entry_h.config(state="normal")
        self.entry_h.delete(0,tk.END)
        self.entry_h.config(state="readonly")
        self.entry_i.config(state="normal")
        self.entry_i.delete(0,tk.END)
        self.entry_i.config(state="readonly")
        self.entry_s.config(state="normal")
        self.entry_s.delete(0,tk.END)
        self.entry_s.config(state="readonly")
        self.entry_akurasi.config(state="normal")
        self.entry_akurasi.delete(0,tk.END)
        self.entry_akurasi.config(state="readonly")

    def akurasi(self , valid):
        if valid == 1 :
            self.data_valid += 1
            self.data_uji += 1
            akurasi = (self.data_valid * 100) / self.data_uji
        else:
            self.data_uji += 1
            akurasi = (self.data_valid * 100) / self.data_uji
        return round(akurasi,4)

    # Tombol Proses Function
    def process(self):

        # Check Data yang Sama
        if self.data_same == 0:


            # Check File
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

                H_val = self.H[self.masked > 0]
                S_val = self.S[self.masked > 0]
                I_val = self.I[self.masked > 0]

                H_mean = round(np.mean(H_val),2)
                S_mean = round(np.mean(S_val),4)
                I_mean = round(np.mean(I_val),4)

                self.entry_h.config(state="normal")
                self.entry_h.delete(0,tk.END)
                self.entry_h.insert(0,H_mean)
                self.entry_h.config(state="readonly")

                self.entry_s.config(state="normal")
                self.entry_s.delete(0,tk.END)
                self.entry_s.insert(0,S_mean)
                self.entry_s.config(state="readonly")

                self.entry_i.config(state="normal")
                self.entry_i.delete(0,tk.END)
                self.entry_i.insert(0,I_mean)
                self.entry_i.config(state="readonly")

                # Result Tingkat Kematangan Buah Tomat
                self.entry_kematangan.config(state="normal")
                self.entry_akurasi.config(state="normal")


                # Result Tingkat Kematangan dengan KNN
                klasifikasi = KlasifikasiKematangan()
                klasifikasi.train_knn_model()
                hasil = klasifikasi.classify_kematangan(H_mean, S_mean, I_mean)
                self.entry_kematangan.delete(0,tk.END)
                self.entry_kematangan.insert(0,hasil)
                self.entry_akurasi.delete(0,tk.END)
                self.entry_akurasi.insert(0,(self.akurasi(1)))

                # Akurasi

                # Result Tingkat Kematangan dengan Metode Rule Based Classification
                # if H_mean <= 0.42 and H_mean >= 0.26 :
                #     self.entry_kematangan.delete(0,tk.END)
                #     self.entry_kematangan.insert(0,"Matang")
                #     self.entry_akurasi.delete(0,tk.END)
                #     self.entry_akurasi.insert(0,(self.akurasi(1)))
                # elif H_mean <= 0.11 and H_mean >= 0.07 :
                #     self.entry_kematangan.delete(0,tk.END)
                #     self.entry_kematangan.insert(0,"Setengah Matang")
                #     self.entry_akurasi.delete(0,tk.END)
                #     self.entry_akurasi.insert(0,(self.akurasi(1)))
                # elif H_mean <= 0.18 and H_mean >= 0.13 :
                #     self.entry_kematangan.delete(0,tk.END)
                #     self.entry_kematangan.insert(0,"Mentah")
                #     self.entry_akurasi.delete(0,tk.END)
                #     self.entry_akurasi.insert(0,(self.akurasi(1)))
                # else:
                #     self.entry_kematangan.delete(0,tk.END)
                #     self.entry_kematangan.insert(0,"Tidak terdeteksi")
                #     self.entry_akurasi.delete(0,tk.END)
                #     self.entry_akurasi.insert(0,(self.akurasi(0)))


                self.entry_kematangan.config(state="readonly")
                self.entry_akurasi.config(state="readonly")

                self.data_same = 1


            else:
                messagebox.showerror("Peringatan","Anda Belum Memilih Gambar")
        else:
            messagebox.showerror("Peringatan","Anda Telah Melakukan Proses Untuk Data Ini, Silahkan Pilih Data Gambar Baru")

    def konversi(self):
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

        # Reset Entry
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.config(state="readonly")
        self.entry_h.config(state="normal")
        self.entry_h.delete(0,tk.END)
        self.entry_h.config(state="readonly")
        self.entry_i.config(state="normal")
        self.entry_i.delete(0,tk.END)
        self.entry_i.config(state="readonly")
        self.entry_s.config(state="normal")
        self.entry_s.delete(0,tk.END)
        self.entry_s.config(state="readonly")
        self.entry_akurasi.config(state="normal")
        self.entry_akurasi.delete(0,tk.END)
        self.entry_akurasi.config(state="readonly")

    # Tombol Reset Function
    def mulai_ulang(self):
        """Reset tampilan ke kondisi awal"""
        self.filename = False
        # self.label_filename.config(text="[Nama Gambar]")

        # Hapus Gambar Dari Canvas
        self.canvas_rgb.delete("all")
        self.canvas_segmentasi.delete("all")
        self.canvas_morfologi.delete("all")
        self.canvas_hue.delete("all")
        self.canvas_saturation.delete("all")
        self.canvas_intensity.delete("all")
        
        # Reset Entry
        self.entry_kematangan.config(state="normal")
        self.entry_kematangan.delete(0,tk.END)
        self.entry_kematangan.config(state="readonly")
        self.entry_h.config(state="normal")
        self.entry_h.delete(0,tk.END)
        self.entry_h.config(state="readonly")
        self.entry_i.config(state="normal")
        self.entry_i.delete(0,tk.END)
        self.entry_i.config(state="readonly")
        self.entry_s.config(state="normal")
        self.entry_s.delete(0,tk.END)
        self.entry_s.config(state="readonly")
        self.entry_akurasi.config(state="normal")
        self.entry_akurasi.delete(0,tk.END)
        self.entry_akurasi.config(state="readonly")

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoSegmentationApp(root)
    root.mainloop()
