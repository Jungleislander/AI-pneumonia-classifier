import base64, os, uuid, glob

import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template, redirect, url_for, session
from PIL import Image

UPLOAD_FOLDER = "tmp"
SAMPLE_DIR = "static/samples"

NORMAL= "Normal"
# Initialize Flask app and set secret key
app = Flask(__name__)
app.secret_key = "AVerySecretKey"

# Load trained model
model = tf.keras.models.load_model("models/xray_model.keras")
IMG_SIZE = (180, 180)

# ──────────────────────  Helpers  ──────────────────────
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

# ──────────────────────  Routes  ──────────────────────
@app.route("/", methods=["GET"])
def upload_form():
    prediction = session.pop("prediction", None)
    confidence = session.pop("confidence", None)
    error = session.pop("error", None)
    image_path = session.pop("image_path", None)
    image_data = None
    is_sample = session.pop("is_sample", False)

    # Delete the uploaded image after it's rendered once
    if image_path and os.path.exists(image_path):
        try:
            image_data = encode_image_to_base64(image_path)
            if not is_sample:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
        except Exception as e:
            error = f"Error loading or deleting image: {str(e)}"

    if confidence is not None and prediction is not None:
        if prediction == NORMAL:
            confidence = f"{((1 - float(confidence))*100):.2f}%"
        else: 
            confidence = f"{(float(confidence)*100):.2f}%"

    sample_files = [
        os.path.basename(p) for p in glob.glob(os.path.join(SAMPLE_DIR, "*"))
        if os.path.isfile(p)
    ]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        image_data=image_data,
        sample_files=sample_files,
    )

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    sample = request.form.get("sample")
    
    try:
        # Determine image source
        if file and file.filename != "":
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
        elif sample:
            file_path = os.path.join("static", "samples", sample)
            session["is_sample"] = True
        else:
            session["error"] = "No file uploaded or sample selected."
            return redirect(url_for("upload_form"))

        if file_path is not None and not os.path.exists(file_path):
            session["error"] = "Selected test image not found."
            return redirect(url_for("upload_form"))
       
        
        session["image_path"] = file_path
        image_array = preprocess_image(file_path)
        pred = model.predict(image_array)[0][0]

        session["prediction"] = "Pneumonia Detected" if pred >= 0.5 else "Normal"
        session["confidence"] = f"{pred:.2f}"

    except Exception as e:
        session["error"] = f"Prediction error: {str(e)}"

    return redirect(url_for("upload_form"))

if __name__ == "__main__":
    # Auto-clean stale uploads on startup
    if not os.path.exists(UPLOAD_FOLDER):
         os.mkdir(UPLOAD_FOLDER)
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    app.run()