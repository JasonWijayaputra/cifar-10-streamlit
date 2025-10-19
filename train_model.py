import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# Nonaktifkan GPU jika tidak ada CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Pastikan folder model ada
os.makedirs("model", exist_ok=True)

# ===============================================================
# 1️. LOAD DAN PERSIAPAN DATASET
# ===============================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisasi pixel ke rentang 0–1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encoding label
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# ===============================================================
# 2️. BANGUN MODEL CNN
# ===============================================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

# ===============================================================
# 3️. KOMPILASI DAN TRAINING
# ===============================================================
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Mulai training model...")
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# ===============================================================
# 4️. EVALUASI
# ===============================================================
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Akurasi Data Uji: {test_acc*100:.2f}%")

# ===============================================================
# 5️. SIMPAN MODEL
# ===============================================================
model.save("model/model_cifar10_optimized.h5")
print("📦 Model disimpan di: model/model_cifar10_optimized.h5")
