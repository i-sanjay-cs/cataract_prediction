from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS  # Import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model("cataract_detection_model_with_densenet.h5")
IMAGE_SIZE = (224, 224)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_cataract(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0][0]
    predicted_class = "Cataract Detected" if prediction > 0.5 else "normal"
    confidence = prediction if predicted_class == "cataract" else 1 - prediction
    return predicted_class, confidence

@app.route('/predict-cataract', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    file.save('temp_image.png')

    predicted_class, confidence = predict_cataract('temp_image.png')

    # Convert confidence to Python float
    confidence = float(confidence)

    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run()
