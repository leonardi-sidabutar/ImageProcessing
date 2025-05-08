from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KlasifikasiKematangan:
    def __init__(self):
        self.knn_model = None


    def train_knn_model(self):
        # Data dari tabel hanya menggunakan fitur H
        mentah = [
            [0.15],
            [0.14],
            [0.14],
            [0.15],
            [0.15],
            [0.13],
            [0.16],
            [0.15],
            [0.18],
            [0.14]
        ]
        
        setengah = [
            [0.08],
            [0.10],
            [0.07],
            [0.09],
            [0.08],
            [0.09],
            [0.09],
            [0.11],
            [0.10],
            [0.11]
        ]
        
        matang = [
            [0.32],
            [0.35],
            [0.28],
            [0.24],
            [0.31],
            [0.28],
            [0.42],
            [0.35],
            [0.33],
            [0.26]
        ]
        
        X = np.array(mentah + setengah + matang)
        y = ['Mentah'] * 10 + ['Setengah Matang'] * 10 + ['Matang'] * 10
        
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.knn_model.fit(X, y)
        print("Model KNN telah dilatih.")

    def classify_kematangan(self, h, s, i):
        if self.knn_model is not None:
            pred = self.knn_model.predict([[h]])[0]
            return pred
        return "Model belum dilatih"

# Menggunakan kelas KlasifikasiKematangan
klasifikasi = KlasifikasiKematangan()

# Melatih model KNN
klasifikasi.train_knn_model()

# Menggunakan model untuk klasifikasi kematangan
h = 0.2 # Contoh nilai H
s = 0.4177  # Contoh nilai S
i = 0.5325  # Contoh nilai I

# Menampilkan prediksi kematangan
result = klasifikasi.classify_kematangan(h, s, i)
print(f"Prediksi kematangan: {result}")
