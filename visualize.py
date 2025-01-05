import visualkeras
from tensorflow.keras.models import load_model

def visualize_model(model_path):
    """
    Load a Keras model from the given path and visualize it using Visualkeras.
    
    :param model_path: Path to the saved Keras model file (e.g., .h5 format).
    """
    try:
        # Load the model
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        
        # Visualize the model
        visualkeras.layered_view(model, legend=True, scale_xy=5).show()
        print("Model visualization generated successfully!")
    except Exception as e:
        print(f"Error loading or visualizing the model: {e}")

if __name__ == "__main__":
    # Replace 'your_model.h5' with the path to your Keras model file
    model_path = input("./model/npk-classifier-v3.keras")
    visualize_model(model_path)
