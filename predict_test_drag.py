# Generic imports
import numpy as np

# Custom imports
from params  import *
from network import *

# Load model
print('Loading model from disk')
model = load_h5_model(model_h5)

# Load image dataset
imgs, n_imgs, height, width, n_channels = load_img_dataset(input_dir_test,
                                                           downscaling,
                                                           color)

# Load solutions dataset
sols, n_sols = load_drag_lift_dataset(sol_dir_test, n_outputs)

# Check consistency
if (n_imgs != n_sols):
    print('Error: I found',n_imgs,'image files and',n_sols,'solutions')
    quit(0)

# Make prediction and compute error
predict_drag, rel_error, avg_error = predict_scalars(model, imgs, sols)

# Plot error
print('Plotting errors')
print('Average relative error = '+str(avg_error))
plot_relative_errors(predict_drag, rel_error, 'drag_relative_error')
