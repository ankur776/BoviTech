# train_model.py - Train Cow/Buffalo Classifier Locally
# Run: python train_model.py (generates .h5 model for app.py)
# Requires: Kaggle API setup and internet for dataset download

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from kaggle.api.kaggle_api_extended import KaggleApi

print(f"[{datetime.now()}] TensorFlow version: {tf.__version__} (Pinned to 2.13 for compatibility)")

# Skip if model already exists (for quick re-runs)
if os.path.exists('cow_buffalo_model.h5'):
    print(f"[{datetime.now()}] Model 'cow_buffalo_model.h5' already exists—skipping training.")
    exit(0)

# GPU setup (optional, improves speed)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[{datetime.now()}] GPU detected and configured.")
    except RuntimeError as e:
        print(f"[{datetime.now()}] GPU config warning: {e}")

# Dataset paths
train_dir = './train'
dataset_slug = 'raghavdharwal/cows-and-buffalo-computer-vision-dataset'

# Auto-download dataset from Kaggle if missing
if not os.path.exists(train_dir):
    print(f"[{datetime.now()}] 'train' folder missing. Downloading dataset from Kaggle...")
    try:
        api = KaggleApi()
        api.authenticate()  # Uses ~/.kaggle/kaggle.json
        api.dataset_download_files(dataset_slug, path='.', unzip=True, force=True)
        print(f"[{datetime.now()}] Dataset downloaded and unzipped! (Check 'train/' for buffalo/ and cow/ folders)")
        if not os.path.exists(train_dir):
            raise FileNotFoundError("'train' folder not created after download—check Kaggle ZIP structure.")
    except Exception as e:
        print(f"[{datetime.now()}] Download failed: {str(e)}")
        print("Manual fix: Download ZIP from https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset")
        print("Accept rules, unzip, and place contents in './train/' (with subfolders 'buffalo/' and 'cow/').")
        exit(1)

print(f"[{datetime.now()}] Using dataset from: {train_dir}")

# Image parameters (matches MobileNetV2)
img_size = (224, 224)
batch_size = 32  # Reduce to 16 if low RAM

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80/20 train/val split
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print(f"[{datetime.now()}] Training samples: {train_generator.samples}, Validation: {val_generator.samples}")
print(f"[{datetime.now()}] Classes found: {train_generator.class_indices}")  # e.g., {'buffalo': 0, 'cow': 1}

# Build model using Functional API (avoids Sequential shape issues)
input_tensor = Input(shape=img_size + (3,))  # Explicit input: (224, 224, 3)
base_model = MobileNetV2(input_tensor=input_tensor, include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Prevent overfitting
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Training
steps_per_epoch = train_generator.samples // batch_size
val_steps = val_generator.samples // batch_size
epochs = 15  # Adjust if needed (10-20 is good)

print(f"[{datetime.now()}] Starting training for {epochs} epochs...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_steps,
    verbose=1  # Shows progress bars
)

# Evaluate on validation set
print(f"[{datetime.now()}] Evaluating model...")
val_loss, val_accuracy = model.evaluate(val_generator, steps=val_steps, verbose=0)
print(f"[{datetime.now()}] Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Save model as .h5 (compatible format for app.py)
model.save('cow_buffalo_model.h5', save_format='h5')
print(f"[{datetime.now()}] Model saved as 'cow_buffalo_model.h5' (ready for Streamlit!)")

# Save class names as list (for easy indexing in app.py)
class_indices = train_generator.class_indices
class_names = sorted(class_indices, key=class_indices.get)  # e.g., ['buffalo', 'cow']
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)  # Save as list: ["buffalo", "cow"]
print(f"[{datetime.now()}] Class names saved: {class_names}")

# Save history for optional plots
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print(f"[{datetime.now()}] Training history saved as 'training_history.pkl'.")

print(f"[{datetime.now()}] ✅ Training complete! Now run 'streamlit run app.py' to test locally.")
print("For Cloud: Commit .h5, .json, .pkl to GitHub (use git lfs track '*.h5').")
