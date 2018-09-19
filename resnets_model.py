import numpy as np
from keras import layers
from keras.layers import Input,Add, Dropout, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, save_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from cnn_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)

# build identity block.
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Fig.3.

    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_c_prev).
        f -- integer, specifying the shape of the middle CONV's window for the main path.
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path.
        stage -- integer, used to name the layers, depending on their position in the network.
        block -- string/character, used to name the layers, depending on their posotion in the netowrk.

    Returns:
        X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C).
    """

    # Define name basis.
    drop_name_base = "dropout" + str(stage) + block + "_branch"
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve filters.
    F1, F2, F3 = filters

    # Save input value for later to add it back to main path.
    X_shortcut = X

    # First component of main path.
    X = Conv2D(filters=F1, kernel_size=1, strides=1, padding="valid", name=conv_name_base+"2a")(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    #X = Dropout(rate=0.15, name=drop_name_base+"2a")(X)
    X = Activation("relu")(X)

    # 2nd compnent of main path.
    X = Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base+"2b")(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    #X = Dropout(rate=0.15, name=drop_name_base+"2b")(X)
    X = Activation("relu")(X)

    # 3rd component of main path.
    X = Conv2D(filters=F3, kernel_size=1, strides=1, padding="valid", name=conv_name_base+"2c")(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)

    # Add shortcut to the main path and pass it thorugh a RELU activation fn.
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X

def conv_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Fig.4.

    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev).
        f -- integer, specifying the shape of the middle CONV layer's window for main path.
        filters -- python list of integers, defining the number of filters in the CONV layer of the main path.
        stage -- integer, used to name the layers, depending on their position in the network.
        block -- string/character, used to name the layers, depending on their position in the network.
        s -- integer, specifying the stride to be used.

    Returns:
        X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C).
    """

    # Defining name basis.
    drop_name_base = "dropout" + str(stage) + block + "_branch"
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters.
    F1, F2, F3 = filters

    # Save the input value for shortcut.
    X_shortcut = X

    # First component of main path.
    X = Conv2D(F1, kernel_size=1, strides=s, padding="valid", name=conv_name_base+"2a")(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2a")(X)
    #X = Dropout(rate=0.15, name=drop_name_base+"2a")(X)
    X = Activation("relu")(X)

    # Second component of main path.
    X = Conv2D(F2, kernel_size=f, strides=1, padding="same", name=conv_name_base+"2b")(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2b")(X)
    #X = Dropout(rate=0.15, name=drop_name_base+"2b")(X)
    X = Activation("relu")(X)

    # Third component of main path.
    X = Conv2D(F3, kernel_size=1, strides=1, padding= "valid", name=conv_name_base+"2c")(X)
    X = BatchNormalization(axis=3, name=bn_name_base+"2c")(X)
    
    # Shortcut path.
    X_shortcut = Conv2D(F3, kernel_size=1, strides=s, padding="valid", name=conv_name_base+"1")(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+"1")(X_shortcut)
    
    # Add shortcut back to main path and pass it through a RELU activation.
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    
    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular Resnet50, the architecture is:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL ->TOPLAYER. 
        
    Arguments:
        input_shape -- shape of the images of the dataset.
        classes -- integer, number of classes.

    Returns:
        model -- a Model() instance in Keras.
    """

    # Define the input as a tensor with shape input_shape.
    X_input = Input(shape=input_shape)

    # Zero-padding.
    X = ZeroPadding2D(padding=3)(X_input)

    # Stage 1.
    X = Conv2D(64, kernel_size=7, strides=2, name="conv1")(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=3, strides=2)(X)

    # Stage 2.
    X = conv_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # Stage 3.
    X = conv_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4.
    X = conv_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5.
    X = conv_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # Average Pooling.
    X = AveragePooling2D(pool_size=2, padding="valid", name="avg_pool")(X)
        
    # Output layer.
    X = Flatten()(X)
    X = Dense(units=classes, activation="softmax", name="fc"+str(classes))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model

def main():
    X_train_orig, Y_train_orig, _, _,  classes = load_dataset()

    # Normalise image vectors.
    #X_train = preprocess_input(X_train_orig)
    X_train = X_train_orig/255
    # Convert training and test labels to one hot matrices.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T

    model = ResNet50(input_shape=(64, 64, 3), classes=len(classes))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=X_train, y=Y_train, batch_size=32, epochs=20)
    save_model(model, "resnets_signs_model.h5")

    #model.summary()
    plot_model(model, to_file="sign_model.png")
    SVG(model_to_dot(model).create(prog="dot", format="svg"))

if __name__ == '__main__':
    main()
    