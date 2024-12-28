import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Build the CNN for one-class anomaly detection (Autoencoder)
input_img = Input(shape=(128, 128, 3))  # Input image shape

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer

# Define the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# Prepare the dataset (only the target class)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize the image
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load only the positive class (rice leaf images)
dataset_path = 'dataset_rice_leaf'  # Path to target class images
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The directory {dataset_path} does not exist.")

train_set = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,  # No labels since it's a one-class model
)

# Debugging dataset
print(f"Found {train_set.samples} images in the dataset.")
if train_set.samples == 0:
    raise ValueError("No images found in the dataset. Check the dataset path and structure.")

# Verify image loading
for batch in train_set:
    print(f"Loaded batch shape: {np.array(batch).shape}")
    break  # Check one batch only

# Train the autoencoder model
steps_per_epoch = max(1, train_set.samples // train_set.batch_size)

try:
    autoencoder.fit(
        train_set,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
    )
except ValueError as e:
    print(f"Error during training: {e}")
    raise

# Save the autoencoder model
autoencoder.save('rice_leaf_autoencoder.h5')
print("Autoencoder model saved as 'rice_leaf_autoencoder.h5'")

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'rice-leaf-autoencoder.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved as {tflite_model_path}')
