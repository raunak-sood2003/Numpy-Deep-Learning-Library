import numpy as np
from urllib import request
import gzip
import pickle
import os

# MNIST dataset from this website: "http://yann.lecun.com/exdb/mnist/"

filenames = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist(dir_path):
    '''
    Downloads MNIST data to path (helper function)
    
    :param dir_path: (str) path to desired directory
    
    :returns VOID
    
    '''
    url = "http://yann.lecun.com/exdb/mnist/"
    for name in filenames:
        print("Downloading %s..." % name[1])
        request.urlretrieve(url + name[1], name[1])
        os.rename('./' + name[1], dir_path + '/' + name[1])
    print("Download complete.")

def save_mnist(dir_path):
    '''
    Saves MNIST data to path (helper function)
    
    :param dir_path: (str) path to desired directory
    
    :returns VOID
    
    '''
    mnist = {}
    for name in filenames[:2]:
        with gzip.open(dir_path + '/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    for name in filenames[2:]:
        with gzip.open(dir_path + '/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    os.rename('./' + 'mnist.pkl', dir_path + '/' + 'mnist.pkl')
    print("Save complete.")

def load_data(dir_path, download):
    '''
    Downloads MNIST data to Path and loads training/testing data
    
    :param dir_path: (str) path to desired directory
    :param download: (bool) indicates whether or not to download data
        (NOTE: this should be false if data was already downloaded to the path)
    
    :returns (numpy.ndarray tuple) (X_train, y_train, X_test, y_test)
        X_train.shape == (60,000, 28, 28)
        y_train.shape == (60,000,)
        X_test.shape == (10,000, 28, 28)
        y_test.shape == (10,000,)
    
    '''
    if download:
        download_mnist(dir_path)
        save_mnist(dir_path)
    with open(dir_path + '/' + 'mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]