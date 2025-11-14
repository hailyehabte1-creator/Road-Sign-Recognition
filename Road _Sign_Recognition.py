# -------------------------------------------------------------
# INSTALL DEPENDENCIES & UPLOAD KAGGLE TOKEN
# -------------------------------------------------------------
!pip install -q kaggle tensorflow tensorflow-addons gradio scikit-learn matplotlib seaborn

from google.colab import files
print("⬆️ Upload your kaggle.json file now.")
files.upload()  # Upload kaggle.json downloaded from Kaggle account

# -------------------------------------------------------------
# SETUP KAGGLE API
# -------------------------------------------------------------
import os

# Create kaggle folder and move token
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("Kaggle API is set up successfully.")

# -------------------------------------------------------------
# DOWNLOAD GTSRB DATASET
# -------------------------------------------------------------
!kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# Unzip dataset
!unzip -q gtsrb-german-traffic-sign.zip -d gtsrb

print("Dataset downloaded and extracted.")
import tensorflow as tf
import pandas as pd
import numpy as np
import os

IMG_SIZE = 224
BATCH_SIZE = 32

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "gtsrb/Train",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Load test dataset manually as it often has a flat structure and .ppm files
# and image_dataset_from_directory might not handle it directly.

# Function to parse image and label
def parse_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

# Read Test.csv to get file paths and labels
test_df = pd.read_csv("gtsrb/Test.csv")
test_image_paths = [os.path.join("gtsrb", test_df['Path'][i]) for i in range(len(test_df))]
test_labels = test_df['ClassId'].values

# Create a TensorFlow dataset from paths and labels
test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_ds = test_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("Number of classes:", NUM_CLASSES)
print("Class names:", class_names[:10], "...")
print(f"Successfully loaded {tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE} training images.")
print(f"Successfully loaded {tf.data.experimental.cardinality(test_ds).numpy() * BATCH_SIZE} test images.")
# -------------------------------------------------------------
# CREATE VALIDATION SPLIT
# -------------------------------------------------------------
VAL_SPLIT = 0.2

train_size = int((1 - VAL_SPLIT) * len(train_ds))
val_size = len(train_ds) - train_size

train_data = train_ds.take(train_size)
val_data = train_ds.skip(train_size)

print("Train batches:", train_size)
print("Val batches:", val_size)
# -------------------------------------------------------------
# CREATE VALIDATION SPLIT
# -------------------------------------------------------------
VAL_SPLIT = 0.2

train_size = int((1 - VAL_SPLIT) * len(train_ds))
val_size = len(train_ds) - train_size

train_data = train_ds.take(train_size)
val_data = train_ds.skip(train_size)

print("Train batches:", train_size)
print("Val batches:", val_size)
# -------------------------------------------------------------
# DATA AUGMENTATION & PREFETCHING
# -------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

train_data = train_data.map(lambda x, y: (data_aug(x, training=True), y))

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)
# -------------------------------------------------------------
# BUILD TRANSFER LEARNING MODEL
# -------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# -------------------------------------------------------------
# INITIAL TRAINING
# -------------------------------------------------------------
EPOCHS = 10

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)
# -------------------------------------------------------------
# FINE-TUNE MODEL
# -------------------------------------------------------------
base_model.trainable = True  # unfreeze full model

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS_FINE = 8

history_fine = model.fit(
    train_data,
    epochs=EPOCHS_FINE,
    validation_data=val_data
)
# -------------------------------------------------------------
# EVALUATE MODEL
# -------------------------------------------------------------
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

print("Classification Report:")
print(classification_report(y_true, y_pred))
# -------------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------------
model.save("gtsrb_mobilenetv2_model")
print("Model saved successfully!")
# -------------------------------------------------------------
# GRADIO DEMO
# -------------------------------------------------------------
import gradio as gr
import numpy as np
from PIL import Image

def predict(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    preds = model.predict(img)
    idx = np.argmax(preds)
    return f"Predicted: {class_names[idx]} (Confidence: {np.max(preds):.3f})"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Road Sign Recognition"
).launch()
