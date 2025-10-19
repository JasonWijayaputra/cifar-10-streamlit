
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# HARUS di sini (paling awal)
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🧠", layout="wide")

# ===============================================================
# 1. LOAD MODEL
# ===============================================================
MODEL_PATH = "model/model_cifar10_optimized.h5"

@st.cache_resource
def load_cifar10_model():
    return load_model(MODEL_PATH)

model = load_cifar10_model()

# ===============================================================
# 2️. LABEL KELAS CIFAR-10
# ===============================================================
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# ===============================================================
# 3️. STREAMLIT UI
# ===============================================================
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown("Upload **satu atau beberapa gambar** untuk diklasifikasikan oleh model CNN.")

# 📁 MULTI FILE UPLOAD
uploaded_files = st.file_uploader(
    "📁 Upload gambar (.jpg, .png, .jpeg)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ===============================================================
# 4️. PREDIKSI DENGAN PROGRESS BAR DAN GRID
# ===============================================================
if uploaded_files:
    num_files = len(uploaded_files)
    st.info(f"🖼️ {num_files} gambar berhasil diunggah. Sedang diproses...")

    # Buat progress bar
    progress_bar = st.progress(0)

    # Tentukan jumlah kolom per baris
    cols_per_row = 3  # bisa diubah menjadi 2, 4 sesuai keinginan

    for i in range(0, num_files, cols_per_row):
        # Ambil batch gambar untuk 1 baris
        batch_files = uploaded_files[i:i+cols_per_row]
        cols = st.columns(len(batch_files))

        for col, uploaded_file in zip(cols, batch_files):
            # Baca gambar
            img = Image.open(uploaded_file).convert("RGB")

            # Preprocessing
            img_resized = img.resize((32, 32))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            # Tampilkan gambar dan hasil prediksi di kolom
            col.image(img, use_column_width=True)
            col.success(f"{uploaded_file.name} → {class_names[class_idx]} ({confidence:.2f}%)")

        # Update progress bar per batch
        progress_bar.progress(min((i + cols_per_row)/num_files, 1.0))

else:
    st.info("Silakan upload satu atau beberapa gambar untuk diklasifikasikan.")
