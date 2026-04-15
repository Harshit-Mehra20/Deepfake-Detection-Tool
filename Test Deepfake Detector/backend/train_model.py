import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

print("Starting training script...")
# Build model
base_model = Xception(weights="imagenet", include_top=False, input_shape=(299,299,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base layers (important)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 🔥 Dummy training (so model becomes usable)
import numpy as np

X_dummy = np.random.rand(10, 299, 299, 3)
y_dummy = np.random.randint(0, 2, 10)

model.fit(X_dummy, y_dummy, epochs=1)

# Save model
model.save("model/deepfake_model.h5")

print("Model saved!")