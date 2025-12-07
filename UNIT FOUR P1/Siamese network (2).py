!pip install -q tensorflow

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

print("Full train shape:", x_train.shape, y_train.shape)
print("Full test shape:", x_test.shape, y_test.shape)

def take_small_subset(images, labels, per_class):

    num_classes = len(np.unique(labels))
    selected_indices = []

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        idx = idx[:per_class]
        selected_indices.extend(idx)

    selected_indices = np.array(selected_indices)
    return images[selected_indices], labels[selected_indices]

# Small train: 100 per digit → 1000 samples
x_train_small, y_train_small = take_small_subset(x_train, y_train, per_class=100)

# Small test: 20 per digit → 200 samples
x_test_small, y_test_small = take_small_subset(x_test, y_test, per_class=20)

print("Small train:", x_train_small.shape, y_train_small.shape)
print("Small test:", x_test_small.shape, y_test_small.shape)

def make_pairs(images, labels):
    """Create positive and negative pairs from images and labels."""
    num_classes = len(np.unique(labels))
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    pairs = []
    pair_labels = []

    for idx in range(len(images)):
        current_image = images[idx]
        label = labels[idx]

        # POSITIVE PAIR (same class)
        pos_idx = np.random.choice(digit_indices[label])
        pos_image = images[pos_idx]

        pairs.append([current_image, pos_image])
        pair_labels.append(1)  # same

        # NEGATIVE PAIR (different class)
        neg_label = np.random.randint(num_classes)
        while neg_label == label:
            neg_label = np.random.randint(num_classes)

        neg_idx = np.random.choice(digit_indices[neg_label])
        neg_image = images[neg_idx]

        pairs.append([current_image, neg_image])
        pair_labels.append(0)  # different

    return np.array(pairs), np.array(pair_labels)

train_pairs, train_labels = make_pairs(x_train_small, y_train_small)
test_pairs, test_labels   = make_pairs(x_test_small, y_test_small)

print("Train pairs:", train_pairs.shape, train_labels.shape)
print("Test pairs:", test_pairs.shape, test_labels.shape)

def build_embedding_network(input_shape=(28, 28, 1)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, (3,3), activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3,3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(32, activation="relu")(x)

    model = keras.Model(inputs, outputs, name="embedding_network")
    return model

embedding_network = build_embedding_network()
embedding_network.summary()

input_shape = (28, 28, 1)

input_a = keras.Input(shape=input_shape, name="image_1")
input_b = keras.Input(shape=input_shape, name="image_2")

embedding_a = embedding_network(input_a)
embedding_b = embedding_network(input_b)

# Absolute difference between embeddings
difference = keras.ops.abs(embedding_a - embedding_b)

x = layers.Dense(32, activation="relu")(difference)
output = layers.Dense(1, activation="sigmoid")(x)

siamese_model = keras.Model(inputs=[input_a, input_b], outputs=output, name="siamese_network")
siamese_model.summary()

x1_train = train_pairs[:, 0]
x2_train = train_pairs[:, 1]

x1_test = test_pairs[:, 0]
x2_test = test_pairs[:, 1]

print("x1_train:", x1_train.shape)
print("x2_train:", x2_train.shape)

siamese_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

history = siamese_model.fit(
    [x1_train, x2_train],
    train_labels,
    validation_data=([x1_test, x2_test], test_labels),
    batch_size=32,
    epochs=50
)

idx1 = 0
idx2 = 8

img1 = x_test_small[idx1:idx1+1]
img2 = x_test_small[idx2:idx2+1]

pred = siamese_model.predict([img1, img2])
similarity = pred.squeeze()  # remove array shape

print("Similarity score:", similarity)
print("Digit1:", y_test_small[idx1], "Digit2:", y_test_small[idx2])

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(img1[0].squeeze(), cmap='gray')
plt.title(f"idx1={idx1}, Digit={y_test_small[idx1]}")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img2[0].squeeze(), cmap='gray')
plt.title(f"idx2={idx2}, Digit={y_test_small[idx2]}")
plt.axis("off")

plt.show()
