from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

# =========================
# LOAD TRAINED MODEL
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_model.h5")

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None

# =========================
# PREPROCESS IMAGE
# Must match training size
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # same as training
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image):
    if model is None:
        return {"error": "Model not loaded"}

    processed = preprocess_image(image)

    prediction = float(model.predict(processed)[0][0])

    is_fake = prediction < 0.5
    confidence = (1 - prediction) if is_fake else prediction

    return {
        "result": "Deepfake" if is_fake else "Real",
        "confidence": round(confidence * 100, 2)
    }

# =========================
# API ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream)
        result = predict_image(image)

        print("Prediction:", result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
