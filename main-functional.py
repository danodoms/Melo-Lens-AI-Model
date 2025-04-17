# -*- coding: utf-8 -*-

import os
import shutil
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Allow TensorFlow to use multiple threads
tf.config.threading.set_inter_op_parallelism_threads(8)  # Set based on your CPU cores
tf.config.threading.set_intra_op_parallelism_threads(8)

# Check CPU usage settings
print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())

# Set seed for reproducibility
random.seed(42)

# Define dataset path
dataset_path = "dataset/watermelon-disease/Augmented Image/Augmented_Image"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

# Create train and val directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Get all class folders inside dataset_path
classes = [d for d in os.listdir(dataset_path)
           if os.path.isdir(os.path.join(dataset_path, d)) and d not in ["train", "val"]]

for cls in classes:
    class_dir = os.path.join(dataset_path, cls)
    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Shuffle images
    random.shuffle(images)

    # Split dataset: 80% train, 20% val
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create class folders in train and val directories
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)

    # Move images to their respective folders
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_path, cls, img))

    for img in val_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(val_path, cls, img))

    # Remove the original class directory if empty
    if not os.listdir(class_dir):
        os.rmdir(class_dir)

print("✅ Dataset successfully split into 'train/' and 'val/'.")

# Load MobileNetV2 without the top layers (pretrained on ImageNet)
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,  # Remove fully connected layers
    weights='imagenet'  # Use pretrained weights
)

# Freeze the base model (so we don’t modify its pretrained weights)
base_model.trainable = False

# Build the model
classifier = Sequential()

# Add the MobileNetV2 feature extractor
classifier.add(base_model)

# Global pooling instead of Flatten() (better for MobileNetV2)
classifier.add(GlobalAveragePooling2D())

# Fully connected layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.5))

# Output layer (match units to the number of classes)
num_classes = len(classes)  # Dynamically set output neurons
classifier.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
classifier.summary()

# Part 2 - Fitting the dataset

# Create ImageDataGenerator for training and testing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and testing data
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    val_path,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Print class indices
label_map = training_set.class_indices
print("Class Labels: ", label_map)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stops training if val_loss doesn't improve for 5 epochs
    restore_best_weights=True
)

# Train the model
classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=20,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size,
    callbacks=[early_stopping]
)

# Part 3 - Save to .keras format
keras_model_path = "model/melon-disease.keras"
classifier.save(keras_model_path)
print(f'✅ Keras model saved as {keras_model_path}')

# Convert the model to TensorFlow Lite (TFLite)
converter = tf.lite.TFLiteConverter.from_keras_model(classifier)

# Enable optimization (reduces model size)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "model/melon-disease.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'✅ TFLite model saved as {tflite_model_path}')
