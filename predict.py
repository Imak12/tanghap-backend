import os
import librosa
import numpy as np
import tensorflow as tf

# 1. SETUP PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "tanghap_model.h5")
SAMPLE_PATH = os.path.join(SCRIPT_DIR, "sample1.wav")

# 2. LOAD MODEL AND LABELS
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

LABELS = ["dry_cough", "non_cough", "persistent_cough", "severe_cough", "wet_cough", "wheezing_cough"]

def predict_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find audio file at: {file_path}")
        
    print(f"Analyzing audio file: {file_path}\n")
    
    # Preprocess audio (16kHz, 2.0s duration)
    audio, sr = librosa.load(file_path, sr=16000, duration=2.0)
    target_length = 16000 * 2
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
        
    # Extract MFCC with n_mfcc=20 and Normalization (katugma ng train.py)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    
    # Expand dimensions for model prediction shape: (1, 20, 63, 1)
    features = np.expand_dims(mfcc, axis=0)
    features = np.expand_dims(features, axis=-1)
    
    # Run prediction
    predictions = model.predict(features, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    
    print("----------------------------------")
    print(f"PREDICTED CLASS: {LABELS[predicted_idx].upper()}")
    print("----------------------------------")
    print("PROBABILITY BREAKDOWN:")
    for idx, label in enumerate(LABELS):
        print(f" - {label}: {predictions[idx]*100:.2f}%")
        print("----------------------------------\n")

predict_audio(SAMPLE_PATH)