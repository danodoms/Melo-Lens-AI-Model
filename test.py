import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to the saved model
model_path = "model/rice_leaf_classifier.keras"

# Load the model
model = tf.keras.models.load_model(model_path)

print(model.summary())
print("Model loaded successfully.")

# Path to the new image
image_path = "hotdog.jpg"

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess the input image to match the model's expected input shape.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size of the image (width, height).
        
    Returns:
        np.array: Preprocessed image array.
    """
    # Load the image
    img = load_img(image_path, target_size=target_size)
    
    # Convert image to array
    img_array = img_to_array(img)
    
    # Scale pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Preprocess the image
input_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_image)

print("Raw predictions:", predictions[0])
