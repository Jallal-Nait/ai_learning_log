import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to 0–1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape if needed (add channel dimension)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Display a figure
    # plt.figure(figsize=(3, 3))
    # plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
    # plt.title(f'Label: {y_train[0]}')
    # plt.axis('off')
    # plt.show()

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),  # 10 classes (digits 0–9)
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Show model architecture
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=5,  # MNIST trains fast — 5 epochs are often enough
        validation_split=0.1,  # Use 10% of training data for validation
        batch_size=32,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    
    # Predict on first test image
    predictions = model.predict(x_test[:1])
    predicted_digit = np.argmax(predictions[0])

    print(f"Model says: {predicted_digit}")
    plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_digit}, True: {y_test[0]}')
    plt.axis('off')
    plt.show()
    
    model.save('mnist_digit_model.keras')