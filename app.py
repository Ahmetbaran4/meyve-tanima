from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
model = load_model("meyve_modeli.keras")

# Etiketleri dış dosyadan yükle
with open("class_indices.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}  # key-value ters çevrilir

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence_percent = None

    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            os.makedirs("test_images", exist_ok=True)
            img_path = os.path.join("test_images", img_file.filename)
            img_file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224)).convert("RGB")
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction_probs = model.predict(img_array, verbose=0)
            confidence = float(np.max(prediction_probs))
            predicted_class = np.argmax(prediction_probs)
            confidence_percent = int(confidence * 100)

            if confidence < 0.7:
                prediction = f"Emin değilim ama tahminim: {class_labels[predicted_class]}"
            else:
                prediction = f"{class_labels[predicted_class]}"

    return render_template("index.html", prediction=prediction, confidence_percent=confidence_percent)

if __name__ == "__main__":
    app.run(debug=True)
