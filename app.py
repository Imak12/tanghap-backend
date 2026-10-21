import os
import io
import gc
import traceback
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dynamic absolute path para sa model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tanghap_model.h5')

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(">>> TANGHAP Model Loaded Successfully! <<<")
    except Exception as e:
        print(f">>> Error loading model: {e} <<<")

CLASS_LABELS = ['dry_cough', 'non_cough', 'persistent_cough', 'severe_cough', 'wet_cough', 'wheezing_cough']

def extract_features_3d(audio_bytes, duration=2.0, sr=16000, n_mfcc=20):
    audio_file = io.BytesIO(audio_bytes)
    # Fast load using librosa
    audio, sr = librosa.load(audio_file, sr=sr, duration=duration, mono=True)
    target_length = int(sr * duration)
    
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
        
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    stacked_channels = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
    return stacked_channels

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "TANGHAP Flask API is running on Render!"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"success": False, "prediction": "unknown", "message": "No audio file provided"}), 400
        
    if model is None:
        return jsonify({"success": False, "prediction": "unknown", "message": "ML Model unavailable"}), 500

    try:
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        features = extract_features_3d(audio_bytes)
        input_tensor = np.expand_dims(features, axis=0)
        
        predictions = model.predict(input_tensor)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        
        predicted_label = CLASS_LABELS[predicted_index]
        accuracy_percent = round(confidence * 100, 2)

        # Clean memory immediately
        del features, input_tensor, audio_bytes
        gc.collect()

        return jsonify({
            "success": True,
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "accuracy_percentage": f"{accuracy_percent}%",
            "raw_scores": {CLASS_LABELS[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(CLASS_LABELS))}
        }), 200

    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return jsonify({"success": False, "prediction": "unknown", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)