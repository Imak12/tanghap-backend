import os
import io
import traceback
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tanghap_model.h5')

print(f"\n>>> Looking for model file at: {MODEL_PATH} <<<")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(">>> TANGHAP ML Model Loaded Successfully! <<<\n")
    except Exception as e:
        print(f">>> Error loading model: {e} <<<\n")
else:
    print(">>> WARNING: tanghap_model.h5 not found! <<<\n")

# EKSAKTONG LABELS AT PAGKAKASUNOD-SUNOD MULA SA TRAIN.PY (6 CLASSES)
CLASS_LABELS = ["dry_cough", "non_cough", "persistent_cough", "severe_cough", "wet_cough", "wheezing_cough"]


def extract_3channel_features(audio_bytes, target_sr=16000, duration=2.0):
    """
    Nag-e-extract ng 3-Channel Stacked Features (MFCC + Delta + Delta2)
    na eksaktong tumutugma sa feature pipeline sa train.py.
    """
    audio_file = io.BytesIO(audio_bytes)
    audio, sr = librosa.load(audio_file, sr=target_sr, duration=duration)
    
    target_length = int(target_sr * duration)  # 16000 * 2.0 = 32000 samples
    actual_duration = len(audio) / target_sr
    
    # Padding o Truncation para maging eksaktong 2.0 seconds
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
        
    # 1. Base MFCC (n_mfcc = 20)
    mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=20)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    
    # 2. Delta & Delta2
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Stack channels -> Output Shape: (20, TimeSteps, 3)
    stacked_channels = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
    
    return stacked_channels, actual_duration


@app.route('/predict', methods=['POST'])
def predict_cough():
    if 'audio' not in request.files:
        return jsonify({
            "success": False, 
            "status": "error",
            "message": "No audio file provided",
            "prediction": "unknown"
        }), 400
        
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    try:
        # 1. Extract 3-Channel Spectrogram Matrix
        features, actual_duration = extract_3channel_features(audio_bytes)

        # 2. Check kung loaded ang model
        if model is None:
            return jsonify({
                "success": False,
                "status": "error",
                "message": "AI Model unavailable. Check tanghap_model.h5 path.",
                "prediction": "unknown"
            }), 500

        # 3. Add Batch Dimension -> Shape: (1, 20, TimeSteps, 3)
        input_data = np.expand_dims(features, axis=0)
        
        predictions = model.predict(input_data)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        predicted_label = CLASS_LABELS[predicted_index]

        return jsonify({
            "success": True,
            "status": "classified",
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "duration": round(actual_duration, 1)
        }), 200

    except Exception as e:
        print("\n==================================================")
        print(">>> DETAILED ML PREDICTION ERROR LOG: <<<")
        traceback.print_exc()
        print("==================================================\n")
        
        return jsonify({
            "success": False,
            "status": "error",
            "message": str(e),
            "prediction": "unknown"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)