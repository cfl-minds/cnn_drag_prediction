# Generic imports
import os
import re
import sys
import math
import numpy as np
import matplotlib
if (sys.platform == 'darwin'):
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Imports with probable installation required
try:
    import skimage
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install scikit-image')
    import skimage

try:
    import keras
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install keras')
    import keras

try:
    import progress.bar
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install progress')
    import progress.bar

from keras.utils               import plot_model
from keras.models              import load_model
from keras.preprocessing.image import img_to_array, load_img

### ************************************************
### Split dataset in training, validation and tests
def split_dataset(dataset, train_size, valid_size, tests_size):
    # Check sizes
    if ((train_size + valid_size + tests_size) != 1.0):
        print('Error in split_dataset')
        print('The sum of the three provided sizes must be 1.0')
        exit()

    # Compute sizes
    n_data     = dataset.shape[0]
    train_size = math.floor(n_data*train_size)
    valid_size = math.floor(n_data*valid_size) + train_size
    tests_size = math.floor(n_data*tests_size) + valid_size

    # Split
    if (dataset.ndim == 1):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size],
                           dataset[train_size:valid_size],
                           dataset[valid_size:])

    if (dataset.ndim == 2):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :],
                           dataset[train_size:valid_size,:],
                           dataset[valid_size:,          :])

    if (dataset.ndim == 3):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :,:],
                           dataset[train_size:valid_size,:,:],
                           dataset[valid_size:,          :,:])

    if (dataset.ndim == 4):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :,:,:],
                           dataset[train_size:valid_size,:,:,:],
                           dataset[valid_size:,          :,:,:])

    return dataset_train, dataset_valid, dataset_tests

### ************************************************
### Load image
def get_img(img_name):
    x = img_to_array(load_img(img_name))

    return x

### ************************************************
### Load and reshape image
def load_and_reshape_img(img_name, height, width, color):
    # Load and reshape
    x = img_to_array(load_img(img_name))
    x = skimage.transform.resize(x,(height,width),
                                 anti_aliasing=True,
                                 mode='constant')

    # Handle color
    if (color == 'bw'):
        x = (x[:,:,0] + x[:,:,1] + x[:,:,2])/3.0
        x = x[:,:,np.newaxis]

    # Rescale
    x = x.astype('float32')/255

    return x

### ************************************************
### Load full image dataset
def load_img_dataset(my_dir, downscaling, color):
    # Count files in directory
    data_files = [f for f in os.listdir(my_dir) if (f[0:5] == 'shape')]
    data_files = sorted(data_files)
    n_imgs     = len(data_files)
    print('I found {} images'.format(n_imgs))

    # Check size of first image
    img    = get_img(my_dir+'/'+data_files[0])
    height = img.shape[0]
    width  = img.shape[1]

    # Declare n_channels
    if (color == 'bw'):  n_channels = 1
    if (color == 'rgb'): n_channels = 3

    # Compute downscaling and allocate array
    height = math.floor(height/downscaling)
    width  = math.floor(width /downscaling)
    imgs   = np.zeros([n_imgs,height,width,n_channels])

    # Load all images
    bar = progress.bar.Bar('Loading imgs  ', max=n_imgs)
    for i in range(0, n_imgs):
        imgs[i,:,:,:] = load_and_reshape_img(my_dir+'/'+data_files[i],
                                             height, width, color)
        bar.next()
    bar.finish()

    return imgs, n_imgs, height, width, n_channels

### ************************************************
### Load drag_lift dataset
def load_drag_lift_dataset(my_dir, n_outputs):
    sol_files = sorted([f for f in os.listdir(my_dir) if f.startswith('shape')])
    n_sols    = len(sol_files)

    sols = np.zeros([n_sols,n_outputs])
    bar  = progress.bar.Bar('Loading labels', max=n_sols)

    for i in range(0, n_sols):
        y = np.loadtxt(my_dir+'/'+sol_files[i], skiprows=1)
        if (n_outputs == 1): sols[i,0]   = y[y.shape[0]-1,1]
        if (n_outputs == 2): sols[i,0:2] = y[y.shape[0]-1,1:3]
        bar.next()
    bar.finish()

    return sols, n_sols

### ************************************************
### Plot relative errors
def plot_relative_errors(predict, error, filename):
    save = np.transpose(predict[:,0])
    if (predict.shape[1] == 2):
        save = np.column_stack((save, np.transpose(predict[:,1])))
    save = np.column_stack((save, np.transpose(error[:,0])))
    if (error.shape[1] == 2):
        save = np.column_stack((save, np.transpose(error[:,1])))

    np.savetxt(filename, save)

    plt.scatter(predict[:,0],error[:,0],c=error[:,0],s=50,cmap='viridis')
    plt.colorbar()
    plt.savefig(filename+'.png')
    plt.show()

### ************************************************
### Plot accuracy and loss as a function of epochs
def plot_accuracy_and_loss(train_model):
    hist       = train_model.history
    train_loss = hist['loss']
    valid_loss = hist['val_loss']
    epochs     = range(len(train_loss))
    np.savetxt('loss',np.transpose([epochs,train_loss,valid_loss]))

    plt.semilogy(epochs, train_loss, 'g', label='Training loss')
    plt.semilogy(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.show()

### ************************************************
### Save keras model to file
def save_keras_model(model):

    model.save('model.h5')

### ************************************************
### Save keras model to file
def evaluate_model_score(model, imgs, sols):
    score = model.evaluate(imgs, sols)
    print('Test accuracy:', score)

    return score

### ************************************************
### Predict scalars from model and compute errors
def predict_scalars(model, imgs, sols):
    # Various stuff
    n_imgs    = len(imgs)
    n_sols    = sols.shape[1]
    predict   = np.zeros([n_imgs,n_sols])
    rel_error = np.zeros([n_imgs,n_sols])
    avg_error = np.zeros([n_sols])

    # Reshape
    h = imgs.shape[1]
    w = imgs.shape[2]
    c = imgs.shape[3]

    # Predict
    for i in range(0, n_imgs):
        img            = imgs[i,:,:,:]
        img            = img.reshape(1,h,w,c)
        predict[i,:]   = model.predict(img)
        rel_error[i,:] = abs((predict[i,:]-sols[i,:])/sols[i,:])
        avg_error[:]  += rel_error[i,:]
    avg_error[:]/=n_imgs

    return predict, rel_error, avg_error

### ************************************************
### Predict images from model and compute errors
def rel_err(x, y, seuil):

    return int(abs(x-y)/(y+1e-6) > seuil)

def fail_count(pred, sol, seuil):
    pred = pred.reshape((pred.shape[0]*pred.shape[1]*pred.shape[2],))
    sol = sol.reshape((sol.shape[0]*sol.shape[1]*sol.shape[2],))
    arr = np.array([rel_err(pred[i], sol[i], seuil) for i in range(len(pred))])

    return np.sum(arr)/len(arr)

def predict_images(model, imgs, sols):
    # Get img shape
    h = imgs.shape[1]
    w = imgs.shape[2]
    c = imgs.shape[3]

    # Various stuff
    n_imgs    = len(imgs)
    predict   = np.zeros([n_imgs,h,w,c])
    abs_error = np.zeros([n_imgs,h,w,c])
    error     = np.zeros([n_imgs, h, w])
    max_error = np.zeros(n_imgs)
    fail      = list()
    mse       = list()

    # Predict
    for i in range(0, n_imgs):
        img                = imgs[i,:,:,:]
        img                = img.reshape(1,h,w,c)
        predict[i,:,:,:]   = model.predict(img)
        abs_error[i,:,:,:] = np.abs(predict[i,:,:,:]-sols[i,:,:,:])
        error[i,:,:]       = np.sum(abs_error[i,:,:,:], axis=2)
        max_error[i]       = np.amax(error[i,:,:])
        fail.append(fail_count(predict[i,:,:,:], sols[i,:,:,:], 0.05))
        mse.append(np.mean(np.square(predict[i,:,:,:]-sols[i,:,:,:])))

    return predict, error, fail, max_error, mse

### ************************************************
### Show an image prediction along with exact image and error
def show_image_prediction(ref_img, predicted_img, error_img, i):
    filename = 'predicted_flow {}_small.png'.format(i)
    fig      = plt.figure(figsize=(5,15))

    ax = fig.add_subplot(3, 1, 1)
    ax.set_title('Reference')
    plt.imshow(ref_img)
    ax = fig.add_subplot(3, 1, 2)
    ax.set_title('Prediction')
    plt.imshow(predicted_img)
    ax = fig.add_subplot(3, 1, 3)
    ax.set_title('Max Error is {}'.format(np.amax(error_img)))
    plt.imshow(error_img/np.amax(error_img), cmap='gray')

    plt.savefig(filename)
    plt.show()

### ************************************************
### Load model
def load_h5_model(model_h5):

    if (not os.path.isfile(model_h5)):
        print('Could not find model file')
        exit()

    model = load_model(model_h5)

    return model

### ************************************************
### Sort data in alphanumeric order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)
