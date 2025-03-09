# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:41:03 2024

@author: rexpogi
"""

# Part 1 : Building a CNN

# Import Keras packages
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Initialize the CNN
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
classifier.add(Dense(units=4, activation='softmax'))

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
    r'dataset/train',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    r'dataset/val',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Print class indices
label_map = training_set.class_indices
print(label_map)

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
tflite_model_path = r'melon-disease.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved as {tflite_model_path}')
