import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Dataset path
dataset_path = r"dataset_rice_leaf/class_name"

# Function to load images from the directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).resize((128, 128))  # Resize images to match model input
            img_array = np.array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
    return np.array(images)

# Load images
images = load_images_from_directory(dataset_path)

# Split into training and validation sets
train_images, val_images = train_test_split(images, test_size=0.1, random_state=42)

# Create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Create generators
train_generator = train_datagen.flow(train_images, batch_size=8)
validation_generator = val_datagen.flow(val_images, batch_size=8)

# Calculate steps_per_epoch and validation_steps based on dataset size
steps_per_epoch = len(train_images) // 8
validation_steps = len(val_images) // 8

# Adjust these values to prevent empty batches
if len(train_images) % 8 != 0:
    steps_per_epoch += 1
if len(val_images) % 8 != 0:
    validation_steps += 1

# Autoencoder Model Definition
input_img = Input(shape=(128, 128, 3))

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
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# Check the first few batches for validity (optional)
def check_for_none_values(generator):
    for i in range(5):  # Check the first few batches
        try:
            batch = next(generator)
            print(f"Batch {i} shape: {batch.shape}")
            if batch is None or len(batch) == 0:
                print(f"Warning: Batch {i} is empty or None.")
                break
        except StopIteration:
            print("Generator exhausted before reaching requested number of batches.")
            break
        except Exception as e:
            print(f"Error with batch {i}: {e}")

check_for_none_values(train_generator)
check_for_none_values(validation_generator)

# Train Autoencoder with verbose output and additional checks for empty batches
try:
    history = autoencoder.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        verbose=1  # Display progress during training
    )
except ValueError as ve:
    print(f"ValueError during model training: {ve}")
except Exception as e:
    print(f"General error during model training: {e}")
    import traceback
    traceback.print_exc()  # Print traceback to help debug

# Save the Model in native Keras format
autoencoder.save('rice_leaf_autoencoder.keras')
print("Autoencoder model saved as 'rice_leaf_autoencoder.keras'")

# Convert the model to TFLite with flex ops enabled (if needed)
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)

    # Enable TF Select for unsupported ops if needed.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_model_path = 'rice-leaf-autoencoder.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f'TFLite model saved as {tflite_model_path}')

except Exception as e:
    print(f"Error during TFLite conversion: {e}")
