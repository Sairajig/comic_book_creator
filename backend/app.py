import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Custom MSE loss function
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Function to build the generator model
def build_generator():
    inputs = Input(shape=(256, 256, 1))  # Input: Grayscale images (256x256x1)
    x = Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same')(inputs)
    x = UpSampling2D(size=(2, 2))(x)
    outputs = Conv2D(3, kernel_size=4, activation='tanh', padding='same')(x)  # Output: RGB images (256x256x3)
    return Model(inputs, outputs)

# Load pre-trained generator model
generator = build_generator()
generator.load_weights('backend/models/generator.h5')  # Load your trained model weights

# GPT-2 setup
tokenizer = GPT2Tokenizer.from_pretrained('backend/models/gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('backend/models/gpt2')

@app.route('/')
def index():
    return jsonify({"message": "Comic Book Creator backend is running!"})

@app.route('/upload_sketch', methods=['POST'])
def refine_sketch():
    try:
        file = request.files['sketch']
        image = Image.open(file).convert('L').resize((256, 256))
        image = np.array(image).reshape(1, 256, 256, 1) / 255.0
        refined = generator.predict(image)
        refined_image = (refined.squeeze() * 255).astype(np.uint8)

        # Convert refined image to base64
        buffered = BytesIO()
        Image.fromarray(refined_image).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"message": "Image refined successfully", "refined_image": img_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_dialogue', methods=['POST'])
def generate_dialogue():
    try:
        prompt = request.json['prompt']
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = gpt_model.generate(inputs['input_ids'], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
