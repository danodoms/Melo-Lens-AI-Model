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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model


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

# Part 1: Load Pretrained Model and Build the Classifier

# Define the input layer
input_layer = Input(shape=(224, 224, 3))

# Load MobileNetV2 pretrained on ImageNet
base_model = MobileNetV2(input_tensor=input_layer, include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)  # Output layer for 4 classes

# Create the Functional API model
classifier = Model(inputs=input_layer, outputs=output_layer)

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
    r'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    r'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices
label_map = training_set.class_indices
print(f"Class indices: {label_map}")

# Part 3: Train the Model with Callbacks

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the model
classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=15,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Part 4: Save the Model in Both Keras and TFLite Formats

# Save the model in the native Keras format
keras_model_path = r'./model/npk-classifier-v2-functional.keras'
classifier.save(keras_model_path, save_format="keras")
print(f'Model saved in native Keras format at {keras_model_path}')

# Convert the model to TensorFlow Lite (TFLite) format
converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = r'./model/npk-classifier-v2-functional.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f'TFLite model saved at {tflite_model_path}')
