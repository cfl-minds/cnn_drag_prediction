# Import stuff
import sys
import math
import keras
import tensorflow as tf

# Additional imports from keras
from keras           import optimizers
from keras.models    import Model
from keras.layers    import Input
from keras.layers    import Conv2D
from keras.layers    import MaxPooling2D
from keras.layers    import Flatten
from keras.layers    import Dense
from keras.layers    import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

# Custom imports
from dataset         import *

### ************************************************
### I/O convolutional layer
def io_conv_2D(x,
               filters     = 8,
               kernel_size = 3,
               strides     = 1,
               padding     = 'same',
               activation  = 'relu'):

    x = Conv2D(filters     = filters,
               kernel_size = kernel_size,
               strides     = strides,
               padding     = padding,
               activation  = activation)(x)

    return x

### ************************************************
### I/O max-pooling layer
def io_maxp_2D(x,
               pool_size = 2,
               strides   = 2):

    x = MaxPooling2D(pool_size = pool_size,
                     strides   = strides)(x)

    return x

### ************************************************
### I/O VGG block
def io_VGG(x,
           nb_fltrs  = 8,
           conv_knl  = 3,
           pool_knl  = 2,
           conv_str  = 1,
           pool_str  = 2,
           n_cv      = 2):

    # n_cv convolutions + pooling
    for cv in range(n_cv):
        x = io_conv_2D(x,
                       filters     = nb_fltrs,
                       kernel_size = conv_knl,
                       strides     = conv_str)
    x = io_maxp_2D(x,
                   pool_size   = pool_knl,
                   strides     = pool_str)

    return x

### ************************************************
### VGG network
def VGG(train_im,
        train_sol,
        valid_im,
        valid_sol,
        test_im,
        h, w,
        channels,
        outputs,
        learning_rate,
        decay,
        batch_size,
        n_epochs):

    # Define VGG16 parameters
    nb_fltrs  = 32
    conv_knl  = 3
    pool_knl  = 2
    conv_str  = 1
    pool_str  = 2
    nb_fcn    = 64

    # Input
    c0 = Input((h, w, channels))

    # VGG blocks
    x = io_VGG(c0,       nb_fltrs,
               conv_knl, pool_knl,
               conv_str, pool_str, 2)
    x = io_VGG(x,        nb_fltrs,
               conv_knl, pool_knl,
               conv_str, pool_str, 2)
    x = io_VGG(x,        2*nb_fltrs,
               conv_knl, pool_knl,
               conv_str, pool_str, 2)
    x = io_VGG(x,        2*nb_fltrs,
               conv_knl, pool_knl,
               conv_str, pool_str, 2)
    x = io_VGG(x,        2*nb_fltrs,
               conv_knl, pool_knl,
               conv_str, pool_str, 2)

    # Flatten
    x = Flatten()(x)

    # Fully-connected and output
    x = Dense(nb_fcn, activation='relu'  )(x)
    x = Dense(1,  activation='linear')(x)

    # Print info about model
    model = Model(inputs=c0, outputs=x)
    model.summary()

    # Set training parameters
    model.compile(loss      = 'mean_squared_error',
                  optimizer = keras.optimizers.Adam(lr    = learning_rate,
                                                    decay = decay))

    early = EarlyStopping(monitor  = 'val_loss',
                          mode     = 'min',
                          verbose  = 0,
                          patience = 10)
    check = ModelCheckpoint('best.h5',
                            monitor           = 'val_loss',
                            mode              = 'min',
                            verbose           = 0,
                            save_best_only    = True,
                            save_weights_only = False)

    # Train network
    with tf.device('/gpu:0'):
        train_model = model.fit(train_im,
                                train_sol,
                                batch_size      = batch_size,
                                epochs          = n_epochs,
                                validation_data = (valid_im, valid_sol),
                                callbacks       = [early, check])

    return(model, train_model)
