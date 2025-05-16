from flask import Flask, request, render_template, jsonify, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# === CLASS NAMES ===
CLASS_NAMES = {
    1: "Eczema", 
    2: "Melanoma", 
    3: "Atopic Dermatitis", 
    4: "Basal Cell Carcinoma (BCC)",
    5: "Melanocytic Nevi (NV)", 
    6: "Benign Keratosis-like Lesions (BKL)",
    7: "Psoriasis, Lichen Planus and related diseases",
    8: "Seborrheic Keratoses and other Benign Tumors",
    9: "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    10: "Warts, Molluscum and other Viral Infections"
}

# === SOLUTIONS MAPPED TO CLASSES ===
SOLUTIONS = {
    1: "Use mild steroid creams and moisturizers. Avoid irritants and consult a dermatologist if symptoms persist.",
    2: "Seek immediate medical attention. Surgery, radiation, or immunotherapy may be required.",
    3: "Use antihistamines and corticosteroids. Keep skin hydrated and avoid allergens.",
    4: "Consult a dermatologist. Surgery or topical treatments like imiquimod may be necessary.",
    5: "Monitor for changes in size or color. Regular dermatologist check-ups recommended.",
    6: "Avoid sun exposure. Use moisturizers and consider cryotherapy or laser treatment if needed.",
    7: "Use topical steroids or immunomodulators. Maintain good skin hygiene and reduce stress.",
    8: "Use keratolytics or cryotherapy. Consult a dermatologist for surgical removal if necessary.",
    9: "Apply antifungal creams and keep skin dry. Avoid sharing personal items.",
    10: "Use salicylic acid, cryotherapy, or laser treatment. Consult a dermatologist if persistent."
}

# === MODEL PATH ===
MODEL_PATH = os.path.join(os.getcwd(), r'final_model.keras')

# === LOAD MODEL ===
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()

# === PREPROCESS IMAGE ===
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# === PREDICT IMAGE ===
def predict_image(img_array):
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0]) + 1  # +1 to align with CLASS_NAMES

        confidence = np.max(predictions[0]) * 100

        # Determine severity based on confidence
        if confidence > 90:
            severity = "Severe"
        elif confidence > 70:
            severity = "Moderate"
        else:
            severity = "Mild"

        class_name = CLASS_NAMES.get(predicted_class, "Unknown Class")
        solution = SOLUTIONS.get(predicted_class, "No solution available for this class.")

        return {
            "class_name": class_name,
            "confidence": f"{confidence:.2f}%",
            "severity": severity,
            "solution": solution
        }
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

# === ROUTES ===
@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    if file:
        try:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            if img_array is None:
                return jsonify({"error": "Failed to process image"})

            result = predict_image(img_array)

            if result:
                return jsonify({
                    "image_url": url_for('static', filename=file.filename),
                    "result": result
                })
            else:
                return jsonify({"error": "Prediction failed"})
        except Exception as e:
            return jsonify({"error": str(e)})

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)


