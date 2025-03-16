import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model

# Build a simple generator
def build_generator():
    inputs = Input(shape=(256, 256, 1))  # Input: Grayscale images (256x256x1)
    x = Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same')(inputs)
    x = UpSampling2D(size=(2, 2))(x)
    outputs = Conv2D(3, kernel_size=4, activation='tanh', padding='same')(x)  # Output: RGB images (256x256x3)
    return Model(inputs, outputs)

# Load custom dataset
def load_data(dataset_path, image_size=(256, 256)):
    sketches = []
    images = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                # Load and preprocess the image
                image = Image.open(os.path.join(root, file)).convert('RGB').resize(image_size)
                sketch = image.convert('L')  # Convert to grayscale (sketch)

                images.append(np.array(image) / 255.0)  # Normalize to range [0, 1]
                sketches.append(np.array(sketch)[:, :, np.newaxis] / 255.0)  # Add channel dimension for grayscale

    return np.array(sketches), np.array(images)

# Training setup
def train_generator_model(dataset_path, output_model_path, epochs=10, batch_size=32):
    # Load dataset
    train_sketches, train_images = load_data(dataset_path)

    # Build and compile the generator model
    generator = build_generator()
    generator.compile(optimizer='adam', loss='mse')

    # Train the model
    generator.fit(train_sketches, train_images, epochs=epochs, batch_size=batch_size)

    # Save the trained model
    generator.save(output_model_path)
    print(f"Model saved to {output_model_path}")

# Main script
if __name__ == "__main__":
    dataset_path = "MANGA109"  # Replace with the actual path to your dataset
    output_model_path = "backend/models/generator.h5"  # Path to save the trained model

    # Train the model
    train_generator_model(dataset_path, output_model_path, epochs=10, batch_size=32)
