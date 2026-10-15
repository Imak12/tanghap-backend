import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# 1. I-load ang na-train na model
model = tf.keras.models.load_model("models/tanghap_model.h5")

# 2. Ang mga eksaktong labels ng iyong ubo
LABELS = ["dry_cough", "non_cough", "persistent_cough", "severe_cough", "wet_cough", "wheezing_cough"]

# 3. Kumuha ng predictions mula sa Test Data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Kung one-hot encoded ang y_test, i-convert sa class index:
y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

# 4. I-print ang Classification Report (Precision, Recall, F1-Score, Accuracy)
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=LABELS))

# 5. I-print ang Confusion Matrix
print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true, y_pred))