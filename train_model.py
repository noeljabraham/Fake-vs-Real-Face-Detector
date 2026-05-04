import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR = "."
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

df = pd.read_csv(CSV_PATH)

# Convert labels: fake = 1, real = 0
df["label"] = df["label"].map({"real": 0, "fake": 1})

# Full image paths
df["filepath"] = df["filepath"].apply(lambda x: os.path.join(DATA_DIR, x))

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["filepath"].values, train_df["label"].values)
)
train_ds = train_ds.map(load_image).shuffle(1000).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["filepath"].values, val_df["label"].values)
)
val_ds = val_ds.map(load_image).batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

os.makedirs("model", exist_ok=True)
model.save("model/fake_real_detector.h5")

print("Model saved successfully.")