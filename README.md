# cnn_drag_prediction

This repository contains a Python code that performs the training of a VGG-like convolutional neural network (CNN) to make drag prediction of arbitrary 2D shapes in low-Reynolds laminar flows (Re = 10). It exploits a custom dataset of random shapes and associated reference drag values for training. The prediction can then be extrapolated to arbitrary shapes, such as geometrical shapes or airfoils.

## Citation

The publication associated to this repository can be found [here](https://arxiv.org/abs/1907.05090).  
If you use this repository for your own research, please consider citing it.

## Usage

The dataset is zipped in the ```MDS0/``` folder to save space on the github repository. You need to unzip it before proceeding further. You will need ```tensorflow``` and ```keras``` installed.

<p align="center">
  <img width="600" alt="network" src="https://user-images.githubusercontent.com/44053700/79682090-d194ac00-821f-11ea-9f52-a209f2e99ac1.png">
</p>

### Training

To train the network described in the paper, just run ```python3 train_network.py```. First, the dataset will be loaded from the ```MDS0``` folder. Then, training will start. Training on a decent GPU card is highly recommended (approx. 25 minutes are required on a single Tesla V100 card). Once learning is over, the best model ```best.h5``` is saved in the working repository. If you wish to make changes, most parameters are accessible in ```params.py```.

<p align="center">
  <img width="600" alt="loss" src="https://user-images.githubusercontent.com/44053700/86446442-430ad380-bd14-11ea-8852-9c2134c87de2.png">
</p>

### Predicting

You can test the produced model on the test subset using ```python3 predict_test_drag.py```. The test subset will be loaded, and the relative errors will be computed. Remember that the multiple sources of randomness (including GPU cards) can lead to slightly different results from one run to another. Same as above, most parameters are accessible in ```params.py```.

<p align="center">
  <img width="600" alt="drag_test" src="https://user-images.githubusercontent.com/44053700/86446550-659cec80-bd14-11ea-9567-36f6664da615.png">
</p>

You can also predict the drag of a shape of your choice by running ```python3 predict_image_drag.py``` (see the parameters at the top of the file). Several shapes are available in the ```imgs``` repository.

<p align="center">
  <img width="600" alt="drag_predictions" src="https://user-images.githubusercontent.com/44053700/64618693-08c8f200-d3e1-11e9-85d8-7eb5f02cc8f3.png">
</p>

## Related repositories

You can find an expanded version of the shape-generation tool contained in this repository [here](https://github.com/jviquerat/bezier_shapes).
