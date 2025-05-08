from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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

# Menggunakan kelas KlasifikasiKematangan
klasifikasi = KlasifikasiKematangan()

# Melatih model KNN
klasifikasi.train_knn_model()

# Menggunakan model untuk klasifikasi kematangan
h = 0.1 # Contoh nilai H
s = 0.858 # Contoh nilai S
i = 0.2771  # Contoh nilai I

# Menampilkan prediksi kematangan
result = klasifikasi.classify_kematangan(h, s, i)
print(f"Prediksi kematangan: {result}")
