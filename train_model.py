"""
train_model.py — Train a CNN on the Kaggle Sugarcane Leaf Disease Dataset
─────────────────────────────────────────────────────────────────────────
Usage:
    python train_model.py --data_dir ./dataset --epochs 25

Saves:
    model/sugarcane_cnn.h5        ← trained weights
    model/class_indices.json      ← {index: class_name} map  ← CRITICAL
"""

import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
SEED     = 42


def build_model(num_classes: int, fine_tune: bool = False):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = fine_tune

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_generators(data_dir: str, batch_size: int):
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.12,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        validation_split=0.2,
    )

    val_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_ds = train_aug.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )

    val_ds = val_aug.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
    )

    return train_ds, val_ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="./dataset")
    p.add_argument("--epochs",    type=int, default=25)
    p.add_argument("--batch",     type=int, default=32)
    p.add_argument("--fine_tune", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"❌  Dataset directory '{args.data_dir}' not found.")
        return

    os.makedirs("model", exist_ok=True)

    print("📂  Loading dataset ...")
    train_ds, val_ds = create_generators(args.data_dir, args.batch)

    # ── CRITICAL: save the exact index→class mapping Keras created ────────────
    idx_to_class = {str(v): k for k, v in train_ds.class_indices.items()}
    with open("model/class_indices.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)

    print(f"✅  Class index map saved → model/class_indices.json")
    print(f"    {idx_to_class}")
    print(f"    Training samples  : {train_ds.samples}")
    print(f"    Validation samples: {val_ds.samples}")

    num_classes = train_ds.num_classes
    print(f"\n🧠  Building MobileNetV2 model ({num_classes} classes) ...")
    model = build_model(num_classes, fine_tune=args.fine_tune)
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(
            "model/sugarcane_cnn.h5",
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
    ]

    print(f"\n🚀  Training for up to {args.epochs} epochs ...")
    
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✅  Best validation accuracy : {val_acc * 100:.2f}%")
    print(f"📦  Model saved  → model/sugarcane_cnn.h5")
    print(f"📋  Class map    → model/class_indices.json")
    print(f"\n⚡  Restart app.py — predictions will now be correct.")


if __name__ == "__main__":
    main()