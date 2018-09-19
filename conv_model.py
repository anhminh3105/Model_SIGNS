#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from IPython import get_ipython
import argparse

np.random.seed(1)
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set the default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
parser = argparse.ArgumentParser()
parser.add_argument("--index_of_image_to_display", default=9, type=int, help="index of the image to display")

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
        n_H0 -- scalar, height of an input image.
        n_W0 -- scalar, width of an input image.
        n_C0 -- scalar, number of channels of the input.
        n_y -- scalar, number of classes.

    Returns:
        X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float".
        Y -- placeholder for the input labels, of shape [None,n_y] and dtype "float".
    """
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(None, n_y), name= "Y")
    return X, Y

def initialise_parameters():
    """
    Initialise weight parameters to build a neural network with tensorflow.
    The shape are:
        W1: [4, 4, 3, 8]
        W2: [2, 2, 8, 16]

    Returns:
        parameters -- a dictionary of tensors containing W1, W2.
    """
    tf.set_random_seed(1)

    W1 = tf.get_variable(name="W1", shape=(4, 4, 3, 8), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable(name="W2", shape=(2, 2, 8, 16), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {
        "W1": W1,
        "W2": W2
    }

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples).
        parameters -- python dictionary containing your parameters "W1, "W2" where the shape are given in initialise_parameters().

    Returns:
        Z3 -- the output of the last LINEAR unit.
    """
    # Retrieve the parameters from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # CONV2D: filters W1, stride of 1, padding "SAME"
    Z1 = tf.nn.conv2d(input=X, filter=W1, strides=[1,1,1,1], padding="SAME")
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding "SAME"
    P1 = tf.nn.max_pool(value=A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding="SAME")
    # CONV2D: filters W2, stride 1, padding "SAME"
    Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=[1,1,1,1], padding="SAME")
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding "SAME"
    P2 = tf.nn.max_pool(value=A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME")
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLYCONNECTED without non-linear activation function (do not call softmax).
    # 6 neurons in output layer.
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=6, activation_fn=None)
    
    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost.

    Arguments:
        Z3 -- output of forward propagation ( output of the last linear unit), of shape (6, number of examples).
        Y -- "true" labels vector placeholder, same shape as Z3.

    Returns:
        cost -- tensor of the cost function.
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
        X_train -- training set of shape (None, 64, 64, 3).
        Y_train -- test set of shape (None, n_y = 6).
        X_test -- training set of shape (None, 64, 64, 3).
        Y_test -- test set of shape (None, n_y = 6).
        learning_rate -- learning rate of the optimisation.
        num_epochs -- number of epochs of the optimisation loop.
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

    Returns:
        train_accuracy -- real number, accuracy on the train set (X_train)
        test_accuracy -- real number, accuracy on the test set (X_test)
        parameters -- parameters learned by the model. They can be used to predict.
    """
    ops.reset_default_graph()   # To be able to rerun the model without overwriting tf variables.
    tf.set_random_seed(1)       # To keep results consistent (tensorflow seed)
    seed = 3                    # To keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []                  # Records the costs during training

    # Create placeholders of the correct shape.
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialise parameters.
    parameters = initialise_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph.
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph.
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimiser. Use an AdamOptimiser that minimises the cost.
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialise all the variables globally.
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph.
    with tf.Session() as sess:
        
        # Run the initialisation.
        sess.run(init)

        # Do the training loop.
        for epoch in range(num_epochs):
            minibatch_cost = 0
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            num_minibatches = len(minibatches)

            for minibatch in minibatches:

                # Select a minibatch.
                (minibatch_X, minibatch_Y) = minibatch
                # **Run the session to execute the optimiser and the cost, the feed_dict should contain a minibatch for (X, Y)
                _, temp_cost = sess.run([optimiser, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every 5 epochs.
            if print_cost == True:
                costs.append(minibatch_cost)
                if epoch % 5 == 0:
                    print("cost after epoch %i: %f." % (epoch, minibatch_cost))

        # Plot the cost.
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iteration (per tens)")
        plt.title("Training Cost with Learning Rate = " + str(learning_rate))
        plt.show()

        # Calculate the correct predictions.
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy of the test set.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy, parameters

def predict(img, parameters):

    x = tf.placeholder(tf.float32, shape=(1,64,64,3))

    Z3 = forward_propagation(x, parameters)

    p = tf.argmax(Z3, 1)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        prediction = sess.run(p, feed_dict={x:img})

    return prediction

def main(argv):
    args = parser.parse_args(argv[1:])

    # Loading the SIGNS dataset.
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
    # Example of a picture.
    '''
    index = args.index_of_image_to_display
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    plt.imshow(X_train_orig[index])
    plt.show()
    '''
    # Pre-process and get info on the shape of the data.
    X_train = X_train_orig/255
    X_test = X_test_orig/255
    Y_train = convert_to_one_hot(Y_train_orig, 6).T # of shape (m, n_y = 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6).T # of shape (n, n_y = 6)
    print("Number of training examples = ", str(X_train.shape[0]))
    print("Number of test examples = ", str(X_test.shape[0]))
    print("X_train.shape: ", str(X_train.shape))
    print("Y_train.shape: ", str(Y_train.shape))
    print("X_test.shape: ", str(X_test.shape))
    print("Y_test.shape: ", str(Y_test.shape))
    # Test create_placeholder():
    '''
    X, Y = create_placeholders(64, 64, 3, 6)
    print("X = ", str(X))
    print("Y = ", str(Y))
    '''
    # Test initialise_parameters():
    '''
    tf.reset_default_graph()
    with tf.Session() as sess:
        parameters = initialise_parameters()
        init = tf.global_variables_initializer()
        sess.run(init)
        print("W1.shape = ", parameters["W1"].shape)
        print("W1[1,1,1] = ", str(parameters["W1"].eval()[1,1,1]))
        print("W2.shape = ", parameters["W2"].shape)
        print("W2 = ", parameters["W2"].eval()[1,1,1])
    '''
    # Test forward_propagation().
    '''
    tf.reset_default_graph()
    np.random.seed(1)
    with tf.Session() as sess:
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialise_parameters()
        Z3 = forward_propagation(X, parameters)
        init = tf.global_variables_initializer()
        sess.run(init)
        a = Z3.eval(feed_dict={X: np.random.randn(2,64,64,3), Y: np.random.randn(2, 6)})
        print("Z3 = ", str(a))
        # Test compute_cost():
        cost = compute_cost(Z3, Y)
        a = sess.run(cost, feed_dict={X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
        print("cost = ", a)
    '''
    _,_,parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=2000)
    image_list = get_img_namelist()
    
    for img in image_list:
        fname = "images/" + img + ".jpg"
        image = np.array(ndimage.imread(fname, flatten=False))
        image = scipy.misc.imresize(image, size=(64,64))
        plt.imshow(image)
        plt.show()
        image = image.reshape((1,64,64,3))
        p = img + ": " + str(predict(image, parameters)[0])
        print(p)

        

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    

