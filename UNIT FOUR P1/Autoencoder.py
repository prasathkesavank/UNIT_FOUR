# 1. Imports & basic setup
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score
)

print("TensorFlow version:", tf.__version__)

# 2. Load MNIST dataset (Colab built-in)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)

# 3. Preprocess data
# Convert to float and normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Add channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

print("Train after reshape:", x_train.shape)
print("Test after reshape :", x_test.shape)
# 4. Define defect-free class and split
# We'll treat digit 0 as the "normal" / defect-free class
NORMAL_CLASS = 0

# Training data: ONLY normal images
x_train_normal = x_train[y_train == NORMAL_CLASS]

# Test data: mixed (normal + anomalies)
x_test_mixed  = x_test
y_test_labels = (y_test != NORMAL_CLASS).astype(int)
# y_test_labels: 0 = normal, 1 = anomaly

print("Normal train samples:", x_train_normal.shape[0])
print("Test samples (mixed):", x_test_mixed.shape[0])
print("Anomaly percentage in test:", y_test_labels.mean() * 100, "%")
# 5. Build a SMALL Dense Autoencoder
input_shape = (28, 28, 1)

inputs = layers.Input(shape=input_shape)
x = layers.Flatten()(inputs)

# Encoder (compress)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
latent = layers.Dense(4, activation="relu", name="latent")(x)  # very small bottleneck

# Decoder (reconstruct)
x = layers.Dense(16, activation="relu")(latent)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = layers.Reshape((28, 28, 1))(x)

autoencoder = models.Model(inputs, outputs, name="dense_autoencoder")
autoencoder.summary()
# 6. Compile & Train the Autoencoder
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mae"   # MAE often works better for anomaly detection
)

history = autoencoder.fit(
    x_train_normal, x_train_normal,
    epochs=30,
    batch_size=128,
    shuffle=True,
    validation_split=0.1
)
# 7. Plot training & validation loss
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.show()
# 8. Compute reconstruction errors

# --- On training NORMAL data (for reference / threshold)
x_train_normal_pred = autoencoder.predict(x_train_normal)
train_errors = np.mean(
    np.abs(x_train_normal_pred - x_train_normal),
    axis=(1, 2, 3)
)

print("Train reconstruction error stats:")
print("  mean:", np.mean(train_errors))
print("  std :", np.std(train_errors))
print("  min :", np.min(train_errors))
print("  max :", np.max(train_errors))

# --- On TEST mixed data ---
x_test_pred = autoencoder.predict(x_test_mixed)
test_errors = np.mean(
    np.abs(x_test_pred - x_test_mixed),
    axis=(1, 2, 3)
)

print("Test reconstruction error stats:")
print("  mean:", np.mean(test_errors))
print("  std :", np.std(test_errors))
print("  min :", np.min(test_errors))
print("  max :", np.max(test_errors))

# 9. Overall ROC-AUC using reconstruction error
roc_auc = roc_auc_score(y_test_labels, test_errors)
print("ROC-AUC (using reconstruction error as score):", roc_auc)

# 12. Threshold 3: Best F1-score for anomaly class (class 1)
best_f1 = 0.0
best_thr_f1 = thresholds[0]

for thr in thresholds:
    y_pred_thr = (test_errors > thr).astype(int)
    f1 = f1_score(y_test_labels, y_pred_thr, pos_label=1)
    if f1 > best_f1:
        best_f1 = f1
        best_thr_f1 = thr

print("Best threshold by F1 (anomaly class):", best_thr_f1)
print("Best F1 for anomaly class:", best_f1)

y_pred_f1 = (test_errors > best_thr_f1).astype(int)

print("\n=== Results with F1-optimal threshold ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test_labels, y_pred_f1))

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_f1, digits=4))
# 13. Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder Anomaly Detection (MNIST)")
plt.legend()
plt.show()
