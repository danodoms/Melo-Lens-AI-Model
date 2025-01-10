# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:41:03 2024
@author: rexpogi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Sequential
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import shutil


# Clear any previous session
tf.keras.backend.clear_session()

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")

# Set random seed for reproducibility
np.random.seed(1337)
tf.random.set_seed(1337)


def prepare_data(input_dir, output_dir, test_size=0.2):
    # Filter out only the actual class directories (exclude 'train' and 'val')
    classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls)) and cls not in ['train', 'val']]

    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        train_images, val_images = train_test_split(images, test_size=test_size, random_state=1337)

        train_dir = os.path.join(output_dir, 'train', cls)
        val_dir = os.path.join(output_dir, 'val', cls)

        # Ensure directories exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for img in train_images:
            img_path = os.path.join(cls_path, img)
            try:
                shutil.copy(img_path, os.path.join(train_dir, img))
            except PermissionError:
                print(f"Permission error while copying {img_path}")
                continue

        for img in val_images:
            img_path = os.path.join(cls_path, img)
            try:
                shutil.copy(img_path, os.path.join(val_dir, img))
            except PermissionError:
                print(f"Permission error while copying {img_path}")
                continue


# Input and output directory
input_dir = r'dataset_balanced'
output_dir = r'dataset_balanced/train_test_split'
prepare_data(input_dir, output_dir)


# Part 1: Load Pretrained Model and Build the Classifier using Sequential API

# Define the model using Sequential API
classifier = Sequential()

# Load MobileNetV2 pretrained on ImageNet without the top layer
classifier.add(MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'))

# Freeze the base model (MobileNetV2)
classifier.layers[0].trainable = False

# Add GlobalAveragePooling2D layer
classifier.add(GlobalAveragePooling2D())

# Add Dense layer with ReLU activation
classifier.add(Dense(128, activation='relu'))

# Add Dropout layer
classifier.add(Dropout(0.5))

# Output layer with softmax activation for 4 classes
classifier.add(Dense(4, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(classifier.summary())


# Part 2: Data Augmentation and Image Generators

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# Only rescale for the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and testing data
training_set = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices
label_map = training_set.class_indices
print(f"Class indices: {label_map}")


# Part 3: Train the Model with Callbacks

# Create TensorBoard logs directory
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# Train the model
classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=1,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size,
    callbacks=[early_stopping, reduce_lr, tensorboard_callback]
)

# # Part 4: Save the Model in Both Keras and TFLite Formats

# # Save the model in the native Keras format
# keras_model_path = r'./model/npk-classifier-v2-sequential.keras'
# classifier.save(keras_model_path, save_format="keras")
# print(f'Model saved in native Keras format at {keras_model_path}')

# # Convert the model to TensorFlow Lite (TFLite) format
# converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
# tflite_model = converter.convert()

# # Save the TFLite model
# tflite_model_path = r'./model/npk-classifier-v2-sequential.tflite'
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)
# print(f'TFLite model saved at {tflite_model_path}')

# To view TensorBoard logs:
# Run in terminal: tensorboard --logdir=logs/fit
