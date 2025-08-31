import tensorflow as tf
import os
import sys

try:
    import tensorflow as tf
except Exception as e:
    print("Could not load TensorFlow", e)
    sys.exit(1)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def load_mnist():
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Print the shapes of the datasets
    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)


if __name__ == "__main__":
    load_mnist()
