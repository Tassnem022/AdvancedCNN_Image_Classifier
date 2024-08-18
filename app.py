# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:57:40 2024

@author: DELL
"""
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('cifar10_cnn_model.h5')

# Dictionary to map numerical labels to CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static/images', 'uploaded_image.jpg')
            file.save(file_path)

            # Preprocess the uploaded image to match CIFAR-10 format
            img = image.load_img(file_path, target_size=(32, 32))
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the class
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred)
            prediction = class_names[predicted_class]
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
