import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. SETTINGS AT PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "datasets")

MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "tanghap_model.h5")
LABELS = ["dry_cough", "non_cough", "persistent_cough", "severe_cough", "wet_cough", "wheezing_cough"]

X = []
y = []

# SPECAUGMENT: FREQUENCY & TIME MASKING FOR AUDIO SPECTROGRAMS
def apply_spec_augment(spectrogram, freq_mask_max=3, time_mask_max=6):
    aug_spec = spectrogram.copy()
    
    # Frequency masking
    num_freq_channels = aug_spec.shape[0]
    f = np.random.randint(0, freq_mask_max)
    f0 = np.random.randint(0, max(1, num_freq_channels - f))
    aug_spec[f0:f0+f, :, :] = 0

    # Time masking
    num_time_steps = aug_spec.shape[1]
    t = np.random.randint(0, time_mask_max)
    t0 = np.random.randint(0, max(1, num_time_steps - t))
    aug_spec[:, t0:t0+t, :] = 0
    
    return aug_spec

# DYNAMIC AUDIO AUGMENTATION ENGINE
def augment_audio(audio, sr):
    augmented_audios = [audio]
    
    # 1. Mild White Noise Insertion
    noise = np.random.randn(len(audio)) * 0.005
    augmented_audios.append(audio + noise)
    
    # 2. Pitch Shift (+1.5 steps)
    try:
        shifted_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.5)
        augmented_audios.append(shifted_up)
    except Exception:
        pass

    # 3. Pitch Shift (-1.5 steps)
    try:
        shifted_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1.5)
        augmented_audios.append(shifted_down)
    except Exception:
        pass

    # 4. Time Stretch (1.08x Speedup)
    try:
        stretched_fast = librosa.effects.time_stretch(audio, rate=1.08)
        target_len = 16000 * 2
        if len(stretched_fast) < target_len:
            stretched_fast = np.pad(stretched_fast, (0, target_len - len(stretched_fast)))
        else:
            stretched_fast = stretched_fast[:target_len]
        augmented_audios.append(stretched_fast)
    except Exception:
        pass

    return augmented_audios

print(f"Checking dataset directory at: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Folder not found at '{DATASET_PATH}'. Please verify folder name!")

# 2. ADVANCED FEATURE EXTRACTION (MFCC + DELTA + DELTA2 + SPECAUGMENT)
for label_index, label in enumerate(LABELS):
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder_path):
        continue
        
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    print(f"Processing '{label}' ({len(files)} base audio files)...")
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            audio, sr = librosa.load(file_path, sr=16000, duration=2.0)
            target_length = 16000 * 2
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            audio_samples = augment_audio(audio, sr)
            
            for sample in audio_samples:
                # 1. Base MFCC
                mfcc = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=20)
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                
                # 2. Delta & Delta2
                delta_mfcc = librosa.feature.delta(mfcc)
                delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                
                # Stack to form 3 Channels: (20, TimeSteps, 3)
                stacked_channels = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
                
                X.append(stacked_channels)
                y.append(label_index)

                # Add SpecAugment version for additional feature robustness
                spec_aug = apply_spec_augment(stacked_channels)
                X.append(spec_aug)
                y.append(label_index)

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError(f"CRITICAL ERROR: No valid audio files found inside '{DATASET_PATH}'!")

# SPLIT DATASET (Stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTotal loaded audio samples (with 3D channels & SpecAugment): {len(X)}")

# CLASS WEIGHT BALANCING
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# 3. HIGH-CAPACITY DEEP CNN ARCHITECTURE
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.35),
    
    # Global Average Pooling Layer
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(LABELS), activation='softmax')
])

optimizer = Adam(learning_rate=0.0008)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# CALLBACKS
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
]

# 4. TRAIN MODEL
print("\nStarting high-accuracy model training...\n")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# 5. EVALUATE MODEL
print("\n================ EVALUATION RESULTS ================")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Overall Model Test Accuracy: {test_acc * 100:.2f}%")
print(f"Overall Model Test Loss:     {test_loss:.4f}\n")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("CLASSIFICATION REPORT PER COUGH TYPE:")
print(classification_report(y_test, y_pred_classes, target_names=LABELS))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred_classes))
print("====================================================\n")

# 6. SAVE MODEL
model.save(MODEL_SAVE_PATH)
print(f"High-accuracy model saved successfully at: '{MODEL_SAVE_PATH}'")