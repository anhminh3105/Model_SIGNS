#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def load_dataset():
    train_dataset = h5py.File("datasets/train_signs.h5", "r")
    print(train_dataset)
    # Get train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    # Get train set labels
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("datasets/test_signs.h5")
    # Get test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    # Get test set labels
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    # Get the list of classes
    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Create a list of random minibatches from (X, Y).

    Arguments:
        X -- input data, of shape (m, n_H, n_W, n_C).
        Y -- true "label" vector (m, n_y).
        mini_batch_size -- size of the minibatches, integer.
        seed -- control the output of the randomness.

    Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # Number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    
    # number of mini_batches of size mini_batch_size in your partitioning.
    num_complete_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_batches):
        batch_start = k*mini_batch_size
        batch_end = batch_start + mini_batch_size
        mini_batch_X = shuffled_X[batch_start:batch_end,:,:,:]
        mini_batch_Y = shuffled_Y[batch_start:batch_end,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_batches*mini_batch_size:m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_batches*mini_batch_size:m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    """
    Convert Y to C numbers of one hot values.

    Arguments:
    Y -- true "label" vector of shape (m, n_y).
    C -- the number of classes in Y

    Returns:
    Y -- true "label" vector of shape (m, n_y*C)
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    #print(Y)
    return Y

def forward_propagation_for_predict(X, parameters):
    """
    Implement the forward propagation for the model LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
        X -- input dataset placeholder, of shape (input_size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3".
                        the shape are given in initialise_parameters

    Returns:
        Z3 -- output of the last linear unit.
    """

    # Retrieve the parameters from the dictionary "parameters".
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.mat_mul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.mat_mul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.mat_mul(W3, A2), b3)

    return Z3

def get_img_namelist(filename="images/img_list.csv"):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        imgs = []
        for row in reader:
            imgs.append(row[0])
    return imgs

'''
def predict(X,parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    with tf.Session() as sess:
        prediction = sess.run(p, feed_dict = {x: X})
    
    return prediction
'''