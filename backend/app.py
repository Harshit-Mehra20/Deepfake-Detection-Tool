from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import sys

# Make sure backend root is on the path so sub-packages resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from db.database import init_db
from routes.auth import auth_bp

app = Flask(__name__)

# Allow requests from:
#   - file:// pages  (browser sends Origin: null)
#   - any localhost port (dev)
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=False)

# =========================
# REGISTER AUTH BLUEPRINT
# =========================
app.register_blueprint(auth_bp)

# =========================
# INITIALISE DATABASE
# =========================
init_db()

# =========================
# LAZY-LOAD TENSORFLOW
# TF is imported inside the route so a broken TF
# installation doesn't prevent Flask from starting.
# =========================
_model = None
_model_loaded = False

def _get_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(os.path.dirname(__file__), "deepfake_model.h5")
        _model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print("Error loading model:", e)
        _model = None
    return _model

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
    model = _get_model()
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
# HEALTH CHECK
# =========================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# =========================
# DETECT API ROUTE
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
