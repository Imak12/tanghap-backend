import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "tanghap_model.h5")
DATASET_PATH = os.path.join(SCRIPT_DIR, "datasets")

# Eksaktong 6 labels base sa train.py
LABELS = ["dry_cough", "non_cough", "persistent_cough", "severe_cough", "wet_cough", "wheezing_cough"]


def extract_features(file_path, duration=2.0, sr=16000, n_mfcc=20):
    """
    Kina-calculate ang 3D channel features (MFCC, Delta, Delta2)
    na eksaktong kapareho ng proseso sa train.py.
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        target_length = int(sr * duration)
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        # 1. Base MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # 2. Delta & Delta2
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Stack to form 3 Channels: (20, TimeSteps, 3)
        stacked_channels = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        return stacked_channels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ===================================================================
# 3. EVALUATION MAIN FUNCTION
# ===================================================================
def evaluate():
    print("\n==================================================")
    print(">>> TANGHAP ML MODEL ACCURACY & MATRIX EVALUATION <<<")
    print("==================================================\n")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    print(f"--> Loading trained model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--> Model Loaded Successfully!\n")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset directory '{DATASET_PATH}' not found!")
        return

    X_eval = []
    y_true = []

    print(f"--> Extracting 3D features from dataset at: {DATASET_PATH}")
    for label_idx, label in enumerate(LABELS):
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder_path):
            print(f"  └─ Skipping missing folder: {label}")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        print(f"  └─ Extracting features from {len(files)} files in '{label}'...")

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                X_eval.append(features)
                y_true.append(label_idx)

    X_eval = np.array(X_eval)
    y_true = np.array(y_true)

    if len(X_eval) == 0:
        print("\n❌ Error: No valid audio features extracted.")
        return

    print(f"\n--> Input Tensor Shape: {X_eval.shape}")
    print("--> Running CNN Model Evaluation...")
    
    predictions = model.predict(X_eval)
    y_pred = np.argmax(predictions, axis=1)

    overall_acc = accuracy_score(y_true, y_pred) * 100

    print("\n==================================================")
    print(f"🎯 TRAINED MODEL OVERALL ACCURACY: {overall_acc:.2f}%")
    print("==================================================\n")

    print("--- CLASSIFICATION REPORT PER COUGH TYPE ---")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'TANGHAP Trained Model - Confusion Matrix\nOverall Accuracy: {overall_acc:.2f}%')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual/True Class')
    plt.tight_layout()

    output_image = os.path.join(SCRIPT_DIR, 'trained_model_confusion_matrix.png')
    plt.savefig(output_image, dpi=300)
    print(f"--> Saved Confusion Matrix plot to: {output_image}")
    plt.show()

if __name__ == '__main__':
    evaluate()