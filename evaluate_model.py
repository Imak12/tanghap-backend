import os
import glob
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. I-load ang trained Keras model
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "tanghap_model.h5")
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Ang eksaktong mga labels ayon sa model
LABELS = ["dry_cough", "non_cough", "persistent_cough", "severe_cough", "wet_cough", "wheezing_cough"]

# 3. Path papunta sa iyong Dataset Folder
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets") 

print("\nExtracting 3D MFCC (MFCC + Delta + Delta2) features from test dataset...")

X_list = []
y_list = []

# Babasahin ang bawat subfolder para sa bawat label
for label_idx, label_name in enumerate(LABELS):
    folder_path = os.path.join(DATASET_DIR, label_name)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        continue

    audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(f"Loading {len(audio_files)} files from class '{label_name}'...")

    for audio_path in audio_files:
        try:
            # I-load ang audio file (3.0 seconds duration, 16kHz sampling rate)
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0)

            # 1. Extract Base 20 MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

            # 2. Extract Delta at Delta-Delta (Velocity & Acceleration)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # 3. Siguraduhing exact 63 time steps ang lapad sa lahat ng 3 channels
            if mfcc.shape[1] > 63:
                mfcc = mfcc[:, :63]
                delta_mfcc = delta_mfcc[:, :63]
                delta2_mfcc = delta2_mfcc[:, :63]
            elif mfcc.shape[1] < 63:
                pad_width = 63 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
                delta_mfcc = np.pad(delta_mfcc, ((0, 0), (0, pad_width)))
                delta2_mfcc = np.pad(delta2_mfcc, ((0, 0), (0, pad_width)))

            # 4. Stack channels -> (20, 63, 3)
            stacked_features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)

            X_list.append(stacked_features)
            y_list.append(label_idx)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

X_test = np.array(X_list)  # Target Shape: (N, 20, 63, 3)
y_test = np.array(y_list)

print(f"\nTotal test samples processed: {len(X_test)}")
print(f"Feature Matrix Shape: {X_test.shape}")

if len(X_test) > 0:
    print("\nEvaluating model predictions...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate overall accuracy percentage
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"\n==========================================")
    print(f" OVERALL MODEL TEST ACCURACY: {accuracy:.2f}%")
    print(f"==========================================\n")

    # Detailed Classification Report
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))

    # Confusion Matrix
    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))
else:
    print("No valid audio files were loaded. Please check your DATASET_DIR path.")