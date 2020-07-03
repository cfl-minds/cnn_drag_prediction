# Generic imports
import time
import numpy as np

# Custom imports
from   params  import *
from   dataset import *
from   network import *

# Handle GPUs
if (train_with_gpu):
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Show the devices in use
print('########### Training drag prediction network ###########')
print('')
print("Devices in use:")
cpus = tf.config.experimental.list_physical_devices('CPU')
for cpu in cpus:
    print("Name:", cpu.name, "  Type:", cpu.device_type)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
print('')

# Load images
imgs, n_imgs, height, width, n_channels = load_img_dataset(input_dir,
                                                           downscaling,
                                                           color)

# Load solutions
sols, n_sols = load_drag_lift_dataset(sol_dir, n_outputs)
sols         = sols[:,0:n_outputs]

# Check consistency
if (n_imgs != n_sols):
    print('Error: I found',n_imgs,'image files and',n_sols,'solutions')
    quit(0)

# Split data into training, validation and testing sets
(imgs_train,
 imgs_valid,
 imgs_tests) = split_dataset(imgs, train_size, valid_size, tests_size)
(sols_train,
 sols_valid,
 sols_tests) = split_dataset(sols, train_size, valid_size, tests_size)

# Print informations
print('Training   set size is', imgs_train.shape[0])
print('Validation set size is', imgs_valid.shape[0])
print('Test       set size is', imgs_tests.shape[0])
print('Input images have size',str(width)+'x'+str(height))

start = time.time()

# Set the network and train it
model, train_model = VGG(imgs_train,
                         sols_train,
                         imgs_valid,
                         sols_valid,
                         imgs_tests,
                         height,
                         width,
                         n_channels,
                         n_outputs,
                         learning_rate,
                         decay,
                         batch_size,
                         n_epochs)

end = time.time()
print("Training time: ",end-start)

# Evaluate score on test set
score = evaluate_model_score(model, imgs_tests, sols_tests)

# Save model
save_keras_model(model)

# Plot accuracy and loss
plot_accuracy_and_loss(train_model)
