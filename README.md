# DL_Assignment2
Assignment 2 on Convolutional Neural Network

## Introduction
The goal of this assignment is twofold: (i) train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters (ii) finetune a pre-trained model just as you would do in many real-world applications


## Requirements
This project requires following 

- Python
- WandB
- Numpy
- PyTorch
- sklearn
- matplotlib


## Understanding the datasets used

The iNaturalist dataset is a large-scale dataset containing images of various plant and animal species, primarily collected from the iNaturalist platform

Size of images are in different dimensions


## Installation

1. Clone the repository:

   bash
   git clone 
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   



## EXPLAINATION OF PY FILES

### train.py
This file contains our model that is trained from scratch and tested on the testing data avalable at test_path.


### pre_train.py
This file contains a pre_trained model where we fine tune the hyperparameter to use it for our test case.


## Usage

 Train the CNN:

   Run the following command to train the neural network:

   bash
   python train.py <list of arguments passed>

   
   You can pass the hyperparameters in command line before training. train_path and test_path are required parameters so we have to pass it to train the model for rest hyperparameter it will take default value.

## INSTRUCTIONS ON HOW TO RUN 

* Create a wandb Account before running train.py and pre_trained.py file.
* Give the api key to your account when prompted.
* install packages as mentioned in the Installation section
  
The following table contains the arguments supported by the train.py file
|Name|	Default Value|	Description|
|:----:| :---: |:---:|
|-wp, --wandb_project	|myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-trp, --train_path|		|Path to training dataset|
|-tsp, --test_path|		|Path to testing dataset|
|-e, --epochs	|5	|Number of epochs to train network.|
|-b, --batch_size|	64	|Batch size used to train  network.|
|-lr, --learning_rate	|0.0001	|Learning rate used to optimize model parameters|
|-a, --activation | mish | choices: ["mish", "relu" , "silu", "gelu"]|
|-bn,--batch_norm | True| choices:[False, True]|
|-da,--data_augmentation|False| choices: [False, True]|
|-d,--dropout|0.2 | dropout value|
|-nf,--num_of_filters| 128 | number of filters in the network|
|-fs,--filter_size | 5 | size of filters|
|-dn,--dense_unit | 512 | Dense layer|
|-fo , --filter_organization | double| choices:["double", "same", "halve"]|


The following table contains the arguments supported by the pre_train.py file rest are the default values.
|Name|	Default Value|	Description|
|:----:| :---: |:---:|
|-wp, --wandb_project	|myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-trp, --train_path|		|Path to training dataset|
|-tsp, --test_path|		|Path to testing dataset|