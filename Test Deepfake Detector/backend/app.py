from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

# Lightweight model (NO TensorFlow heavy usage)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

app = Flask(__name__)
CORS(app)

# ✅ Lightweight model (no OOM)
def create_light_model():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_light_model()

# ✅ Preprocess image (small size → no crash)
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Prediction
def predict_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    is_fake = prediction > 0.5
    confidence = float(prediction if is_fake else 1 - prediction)

    return {
        "result": "Deepfake" if is_fake else "Real",
        "confidence": round(confidence * 100, 2)
    }

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file)

    result = predict_image(image)
    print("Prediction:", result)   # 🔥 DEBUG
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)