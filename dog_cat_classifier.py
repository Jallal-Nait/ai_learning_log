import os
from pathlib import Path
import sys
from typing import Optional

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, utils
except Exception as e:
    print("Could not load TensorFlow", e)
    sys.exit(1)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10


def create_new_model(model_path: Path, train_dir: Path, validation_dir: Path):
    # Load data from directories
    train_dataset = utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="binary",  # binary labels = 0=cat, 1=dog
        seed=123,
    )

    # Load data from /data
    validation_dataset = utils.image_dataset_from_directory(
        validation_dir,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=123,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    )

    model = models.Sequential(
        [
            # Rescaling layer (replaces rescale=1./255)
            layers.Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            # Augmentation
            data_augmentation,
            # CNN layers
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Binary output
        ]
    )

    # Compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

    model.save(model_path)  # Recommended `.keras` format


def classify_dog_or_cate(img_path: Path, model_path: Path) -> Optional[dict]:
    if not model_path.exists():
        print("Could not find model path", model_path)
        return None

    if not img_path:
        print("Could not find image", img_path)
        return None

    img = utils.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

    img_array = utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Load the entire model
    model = tf.keras.models.load_model(model_path)

    # Confirm it's loaded
    model.summary()

    prediction = model.predict(img_array)

    result = {"animal": None, "confidence": prediction[0][0]}

    if prediction[0] > 0.5:
        result["animal"] = "🐶 Dog"
    else:
        result["animal"] = "🐱 Cat"

    return result
