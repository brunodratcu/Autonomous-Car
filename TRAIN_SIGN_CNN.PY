# train_sign_cnn.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIG
# ============================================================

IMG_W = 64
IMG_H = 64
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = "./dataset"
MODEL_PATH = "./models/sign_cnn.h5"
LABELS_PATH = "./labels.txt"

CLASS_NAMES = [
    "stop",
    "yield",
    "left",
    "right",
    "straight",
    "delivery",
    "unknown"
]

# ============================================================
# GARANTIR PASTA DE MODELOS
# ============================================================

os.makedirs("./models", exist_ok=True)

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    brightness_range=(0.8, 1.2),
    shear_range=0.08,
    horizontal_flip=False,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "val"),
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ============================================================
# VALIDAR ORDEM DAS CLASSES
# ============================================================

idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
ordered_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Class indices:", train_gen.class_indices)
print("Ordered labels:", ordered_labels)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for label in ordered_labels:
        f.write(label + "\n")

# ============================================================
# MODELO CNN PEQUENO
# ============================================================

model = models.Sequential([
    layers.Input(shape=(IMG_H, IMG_W, 3)),

    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(len(ordered_labels), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# ============================================================
# TREINO
# ============================================================

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# SALVAR MODELO FINAL
# ============================================================

model.save(MODEL_PATH)
print(f"Modelo salvo em: {MODEL_PATH}")
print(f"Labels salvos em: {LABELS_PATH}")