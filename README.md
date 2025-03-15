``` import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# 🚀 Fungsi untuk upload file
def upload_file():
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    print(f"File '{filename}' berhasil diunggah.")
    return filename

# 🚀 Load image (grayscale mode)
def load_image(path):
    if not os.path.exists(path):  # Cek apakah file ada
        print(f"Error: File '{path}' tidak ditemukan!")
        return None
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: File '{path}' gagal dimuat! Pastikan formatnya benar.")
    return image

# 🚀 1. Citra Negative
def negative_image(image):
    return 255 - image

# 🚀 2. Transformasi Log
def log_transform(image):
    if np.max(image) == 0:
        return image  # Hindari error log(0)
    c = 255 / (np.log(1 + np.max(image)) + 1e-8)
    return (c * np.log(1 + image)).astype(np.uint8)

# 🚀 3. Transformasi Power Law (Gamma Correction)
def power_law_transform(image, gamma=1.0):
    if np.max(image) == 0:
        return image  # Hindari error
    c = 255 / ((np.max(image) + 1e-8) ** gamma)
    return (c * (image ** gamma)).astype(np.uint8)

# 🚀 4. Histogram Equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# 🚀 5. Histogram Normalization
def histogram_normalization(image):
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val == 0:
        return image  # Hindari pembagian dengan nol
    normalized = (image - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)

# 🚀 Tampilkan gambar hasil processing
def display_results(original, results, titles):
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    
    for i in range(len(results)):
        plt.subplot(2, 3, i+2)
        plt.imshow(results[i], cmap='gray')
        plt.title(titles[i])
        plt.axis("off")
    
    plt.show()

# 🚀 Upload gambar
print("Silakan unggah gambar yang ingin diproses...")
image_path = upload_file()

# 🚀 Load gambar
image = load_image(image_path)

# 🚀 Jika gambar ditemukan, lanjutkan proses
if image is not None:
    print("Gambar berhasil dimuat!")

    # Proses gambar dengan metode yang diberikan
    negative = negative_image(image)
    log_transformed = log_transform(image)
    power_law = power_law_transform(image, gamma=0.5)
    hist_eq = histogram_equalization(image)
    hist_norm = histogram_normalization(image)

    # Menampilkan hasil
    results = [negative, log_transformed, power_law, hist_eq, hist_norm]
    titles = ["Negative", "Log Transform", "Power Law", "Hist Equalization", "Hist Normalization"]
    display_results(image, results, titles)

else:
    print("Gagal memproses gambar! Coba unggah ulang dengan format yang benar.")```

--------------------------------------------------------------------------------------------------
    Penjelasan Kondisi Input dan Output:
1. Citra Negatif:
Input: Gambar grayscale.
Output: Warna terang menjadi gelap dan sebaliknya.

2. Transformasi Log:
Input: Gambar grayscale.
Output: Kontras tinggi pada intensitas rendah, cocok untuk gambar gelap.

3. Transformasi Power Law (Gamma Correction):
Input: Gambar grayscale.
Output: Meningkatkan kontras berdasarkan nilai gamma.

4. Histogram Equalization:
Input: Gambar grayscale.
Output: Distribusi intensitas lebih merata, meningkatkan detail pada area gelap/terang.

5. Histogram Normalization:
Input: Gambar grayscale.
Output: Rentang intensitas disesuaikan ke [0, 255] untuk meningkatkan kontras.

6. Konversi RGB ke HSI:
Input: Gambar berwarna.
Output: Gambar dalam format HSI, cocok untuk deteksi warna dan segmentasi.

Cara Menentukan Thresholding pada HSI:
Ekstrak Kanal H (Hue): Digunakan untuk segmentasi warna.
Gunakan Rentang Threshold: Misalnya, untuk mendeteksi warna merah, gunakan nilai H dalam rentang tertentu.
Gunakan Masking: Terapkan threshold untuk memisahkan objek dengan warna yang diinginkan.
