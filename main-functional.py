# -*- coding: utf-8 -*-
"""
Created on Sun Mar 09 2025

@author: rexpogi
"""

import os
import shutil
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
        shutil.move(os.path.join(class_dir, img), os.path.join(train_path, cls, img))
    
    for img in val_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(val_path, cls, img))

    # Remove the original class directory if empty
    if not os.listdir(class_dir):
        os.rmdir(class_dir)

print("✅ Dataset successfully split into 'train/' and 'val/'.")

# Part 1 : Building a CNN
np.random.seed(1337)
classifier = Sequential()

# Input layer
classifier.add(Input(shape=(128, 128, 3)))

# Convolutional and pooling layers
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(16, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(8, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening and fully connected layers
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.5))

# Output layer (match units to the number of classes)
num_classes = len(classes)  # Dynamically set output neurons
classifier.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

# Train the model
classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=20,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size
)

# Part 3 - Convert the model to TensorFlow Lite (TFLite)

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "melon-disease.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'✅ TFLite model saved as {tflite_model_path}')
