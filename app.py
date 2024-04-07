from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('image_classifier_model.h5')

# Define class names
class_names = ['Shoes', 'Sandals', 'Boots']

def preprocess_image(img):
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def classify_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    result = None
    uploaded_image = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            predicted_class, confidence = classify_image(img)
            result = f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%"
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', 'uploads', filename)
            img.save(file_path)
            uploaded_image = url_for('static', filename=f'uploads/{filename}')
        else:
            return "You should upload the picture first"  # Returning a plain string to indicate the error
    return render_template('index.html', result=result, uploaded_image=uploaded_image)

@app.route('/update_label', methods=['POST'])
def update_label():
    if request.method == 'POST':
        data = request.get_json()
        filename = data['filename']
        new_label = data['new_label']
        # Add code to update training data with new label
        # For simplicity, let's assume updating a CSV file or a database
        return jsonify({'message': 'Label updated successfully'})

if __name__ == '__main__':
    app.run(debug=True)
