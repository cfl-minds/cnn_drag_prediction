# Imports
import os
import logging
import tensorflow as tf
import numpy      as np
import random     as rn
import keras      as k

# Data directoriese
dataset_dir       = 'DS0'
input_dir         = dataset_dir+'/shapes'
sol_dir           = dataset_dir+'/drags'
input_dir_test    = dataset_dir+'/shapes_test'
sol_dir_test      = dataset_dir+'/drags_test'

# Model directories
model_dir         = './'
model_h5          = model_dir+'best.h5'

# Image data
img_width         = 256
img_height        = 256
downscaling       = 2
color             = 'bw'

# Dataset data
train_size        = 0.8
valid_size        = 0.1
tests_size        = 0.1

# Learning data
n_outputs         = 1
learning_rate     = 1.0e-3
n_epochs          = 500
decay             = 0.005
batch_size        = 256
network           = 'VGG16'

# Hardware
train_with_gpu    = True

# Set tf verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
logger                             = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set random seeds
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
tf.random.set_seed(1)
np.random.seed(1)
rn.seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                            config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
