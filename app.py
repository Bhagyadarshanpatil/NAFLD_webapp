import io
import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --------------------
# Configuration
# --------------------
# Phase 1: Clinical Model (Tabular)
PHASE1_MODEL = "phase1_nafld_model.pkl"
PHASE1_SCALER = "phase1_scaler.pkl"

# Phase 2: Ultrasound (CNN + FFT)
PHASE2_MODEL = "phase2_cnnFFT_model.keras"
IMG_SIZE = (256, 256)

# Phase 3: Longitudinal (LSTM)
PHASE3_MODEL = "phase3_var_lstm_1.keras"
PHASE3_SCALER = "phase3_scaler.pkl"

# --------------------
# Flask app
# --------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app) 

print("⏳ Initializing NAFLD Clinical Server...")

# --------------------
# Utils / Custom fn for Models
# --------------------
def perform_fft_processing(input_tensor):
    """TensorFlow implementation of FFT preprocessing used inside some saved models."""
    x = tf.squeeze(input_tensor, axis=-1)  # (B, H, W)
    x = tf.cast(x, tf.complex64)
    x = tf.signal.fft2d(x)
    x = tf.signal.fftshift(x)
    x = tf.abs(x)
    x = tf.math.log(1.0 + x)
    x = tf.expand_dims(x, axis=-1)
    return x

# --------------------
# 1. LOAD Phase 1 (Clinical) Model
# --------------------
model_p1 = None
try:
    # Try loading the model if it exists
    if os.path.exists(PHASE1_MODEL):
        model_p1 = joblib.load(PHASE1_MODEL)
        print(f"✅ Phase 1: Clinical Model Loaded ({PHASE1_MODEL})")
    else:
        print(f"⚠️ Phase 1: '{PHASE1_MODEL}' not found. Using Rule-Based Fallback.")
except Exception as e:
    print(f"❌ Phase 1 Error: {e}")

# --------------------
# 2. LOAD Phase 2 (Ultrasound) Model
# --------------------
model_p2 = None
try:
    if os.path.exists(PHASE2_MODEL):
        custom_objects = {"perform_fft_processing": perform_fft_processing}
        # Attempt load with custom objects
        try:
            model_p2 = tf.keras.models.load_model(PHASE2_MODEL, custom_objects=custom_objects, compile=False)
            print("✅ Phase 2: CNN-FFT Model Loaded (with custom_objects)")
        except:
            # Fallback to safe_mode=False
            model_p2 = tf.keras.models.load_model(PHASE2_MODEL, safe_mode=False, compile=False)
            print("✅ Phase 2: CNN-FFT Model Loaded (safe_mode=False)")
    else:
         print(f"⚠️ Phase 2: '{PHASE2_MODEL}' not found.")
except Exception as e:
    print(f"❌ Phase 2 Error: {e}")

# --------------------
# 3. LOAD Phase 3 (LSTM) Model
# --------------------
model_p3, scaler_p3 = None, None
try:
    if os.path.exists(PHASE3_MODEL) and os.path.exists(PHASE3_SCALER):
        model_p3 = tf.keras.models.load_model(PHASE3_MODEL)
        scaler_p3 = joblib.load(PHASE3_SCALER)
        print("✅ Phase 3: LSTM & Scaler Loaded")
    else:
        print(f"⚠️ Phase 3: Model or Scaler not found.")
except Exception as e:
    print(f"❌ Phase 3 Error: {e}")


# --------------------
# PREPROCESSING HELPERS
# --------------------
def preprocess_phase2_image_numpy(image_file):
    img = Image.open(image_file).convert('L')
    img = img.resize(IMG_SIZE)
    img_arr = np.array(img).astype('float32') / 255.0
    spatial_in = np.expand_dims(img_arr, axis=(0, -1))

    # FFT branch
    f = np.fft.fft2(img_arr)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-7)
    if np.max(magnitude) != 0:
        fft_norm = magnitude / np.max(magnitude)
    else:
        fft_norm = magnitude
    fft_in = np.expand_dims(fft_norm, axis=(0, -1))
    return spatial_in, fft_in

def preprocess_phase2_image_spatial_only(image_file):
    img = Image.open(image_file).convert('L')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=(0, -1))


# --------------------
# ROUTES
# --------------------

@app.route('/')
def index():
    return jsonify({
        "status": "NAFLD System Online", 
        "models": {
            "clinical": bool(model_p1),
            "ultrasound": bool(model_p2),
            "longitudinal": bool(model_p3)
        }
    })

# ---------------------------------------------------------
# NEW: Phase 1 Clinical Prediction (Matches patient-detail.js)
# ---------------------------------------------------------
@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    try:
        data = request.json
        print("📥 Received Clinical Data:", data)

        # 1. Parse Input (Handle string/number conversions)
        age = float(data.get('age', 0))
        bmi = float(data.get('bmi', 0))
        gender_val = 1 if data.get('gender', '').lower() == 'male' else 0
        
        # Liver Panel
        alt = float(data.get('alt', 0))
        ast = float(data.get('ast', 0))
        alp = float(data.get('alp', 0))
        albumin = float(data.get('albumin', 0))
        ag_ratio = float(data.get('agRatio', 0))
        total_bil = float(data.get('totalBilirubin', 0))
        direct_bil = float(data.get('directBilirubin', 0))
        total_prot = float(data.get('totalProtein', 0))

        # 2. Predict
        if model_p1:
            # Construct feature vector matching YOUR trained model's order
            # Ensure this list matches exactly how you trained 'clinical_model.pkl'
            features = np.array([[age, gender_val, bmi, alt, ast, alp, albumin, ag_ratio, total_bil, direct_bil, total_prot]])
            
            # Predict
            pred_cls = model_p1.predict(features)[0]
            try:
                probs = model_p1.predict_proba(features)
                risk_score = round(probs[0][1] * 100, 2)
            except:
                risk_score = 0 if pred_cls == 0 else 85 # Fallback if no probability
            
            prediction_text = "NAFLD Detected" if pred_cls == 1 else "No NAFLD"
        
        else:
            # --- FALLBACK / DUMMY LOGIC (Use this if no model is trained yet) ---
            print("Using Rule-Based Logic (No Model Loaded)")
            risk_score = 0
            
            # Simple medical heuristics for demo
            if bmi > 25: risk_score += 20
            if bmi > 30: risk_score += 15
            if alt > 40: risk_score += 25
            if ast > 40: risk_score += 20
            if age > 50: risk_score += 10
            if gender_val == 1: risk_score += 5
            
            # Cap at 99
            risk_score = min(risk_score, 99)
            prediction_text = "NAFLD Detected" if risk_score > 45 else "No NAFLD"

        # 3. Return JSON exactly as frontend expects
        return jsonify({
            "prediction": prediction_text,
            "risk_score": risk_score
        })

    except Exception as e:
        print("Error in /predict_clinical:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# Phase 2: Ultrasound Prediction
# ---------------------------------------------------------
@app.route('/predict_ultrasound', methods=['POST'])
def predict_phase2():
    if not model_p2:
        return jsonify({"error": "Ultrasound model not loaded"}), 500

    # Robust file handling
    file_key = 'images' if 'images' in request.files else 'image'
    if file_key not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files[file_key]
    
    try:
        # Check model inputs to decide preprocessing
        num_inputs = len(model_p2.inputs) if hasattr(model_p2, 'inputs') else 1
        
        f_stream = io.BytesIO(f.read())
        
        if num_inputs == 2:
            # Spatial + FFT
            spatial_in, fft_in = preprocess_phase2_image_numpy(f_stream)
            preds = model_p2.predict([spatial_in, fft_in])
        else:
            # Spatial Only (or internal FFT)
            arr = preprocess_phase2_image_spatial_only(f_stream)
            preds = model_p2.predict(arr)

        # Decode Prediction
        preds_arr = np.array(preds)
        
        # Handle binary vs multi-class
        if preds_arr.shape[1] == 1:
            prob = float(preds_arr[0, 0])
            idx = int(prob >= 0.5)
        else:
            idx = int(np.argmax(preds_arr[0]))
            prob = float(preds_arr[0, idx])

        stages = ["F0 (Healthy)", "F1 (Mild)", "F2 (Moderate)", "F3 (Severe)", "F4 (Cirrhosis)"]
        result_stage = stages[idx] if idx < len(stages) else f"Stage {idx}"

        return jsonify({
            "stage": result_stage,
            "probability": f"{prob:.2f}",
            "raw": preds.tolist()
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# Phase 3: Longitudinal Prediction
# ---------------------------------------------------------
@app.route('/predict_phase3', methods=['POST'])
def predict_phase3():
    if not model_p3 or not scaler_p3:
        return jsonify({"error": "Phase 3 model/scaler missing"}), 500

    try:
        d = request.json
        # Extract features matching training
        raw_feats = [
            d.get('age', 0),
            d.get('male', 0),
            d.get('bmi', 0),
            d.get('hdl', 0),
            d.get('chol', 0),
            d.get('sbp', 0),
            d.get('dbp', 0),
            d.get('smoke', 0),
            d.get('fib4', 0)
        ]
        
        # Reshape & Scale
        feats_arr = np.array([raw_feats])
        scaled_feats = scaler_p3.transform(feats_arr)

        # Prepare LSTM Sequence (1, 6, features)
        TIMESTEPS = 6
        num_features = scaled_feats.shape[1]
        sequence_input = np.zeros((1, TIMESTEPS, num_features), dtype='float32')
        # Insert current visit at the end
        sequence_input[0, -1, :] = scaled_feats

        pred_prob = model_p3.predict(sequence_input)
        risk_score = float(pred_prob[0][0]) if np.ndim(pred_prob) >= 2 else float(pred_prob[0])
        
        risk_label = "High Risk" if risk_score > 0.5 else "Low Risk"
        
        return jsonify({
            "risk_probability": round(risk_score, 4),
            "risk_label": risk_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)