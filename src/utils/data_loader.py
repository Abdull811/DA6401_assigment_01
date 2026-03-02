import numpy as np
from keras.datasets import mnist, fashion_mnist #load the datasets
from sklearn.model_selection import train_test_split # Split the dataset into train and test

# Load MNIST or Fashion MNIST dataset
# And return training and testing images labels: x_train, y_train, x_test, y_test
def load_data(name):
    if name == 'mnist':
      (x_train, y_train),(x_test, y_test) = mnist.load_data()
    elif name == 'fashion_mnist':
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
       print()

    # Normalize x_train, x_test dataset
    x_train = (x_train/255.0).astype(np.float32)
    x_test = (x_test/255.0).astype(np.float32)
    x_train = x_train.reshape(x_train.shape[0], -1) #Reshape train image from 2D to 1D
    x_test = x_test.reshape(x_test.shape[0], -1) # Reshape test image

    # Train and test split
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                            test_size=0.2, random_state = 42)
    return x_train, y_train, x_valid, y_valid, x_test, y_test
