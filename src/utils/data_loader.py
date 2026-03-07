import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def load_data(name):
    if name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Normalize train and test
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Flatten images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Split train into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test
