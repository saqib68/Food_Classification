from flask import Flask, request, render_template, jsonify, url_for
import tensorflow as tf
from PIL import Image
import numpy as np   
import os

app = Flask(__name__, static_folder='static')

# Define the categories
CATEGORIES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"
]

# Load the model (you'll need to replace this with your actual model path)
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    # Resize image to match model's expected sizing
    img = Image.open(image)
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Preprocess the image
        img_array = preprocess_image(file)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = CATEGORIES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 