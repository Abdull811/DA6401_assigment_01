# ASSIGMENT 1: Multi-Layer Perceptron For Image Classification

## Overview 
This assigments a Multi-layer perceptron (MLP) neural network from acratch using numpy for image classification. Experiments are tracked using Weight & Biases for visualization and hyperparameter tuning.

## Repository 
GitHub Repository: [https://github.com/Abdull811/DA6401_assigment_01.git]
Weights & Biases Report: []

## Learniing Objective
- Forward propagation in neural networks
- Backpropagation and gradient computation
- Implementing activations (ReLu, Sigmoid and Tanh) and their derivatives
- Implementing optimization algorithms (Stochastic Gradient Descent (SGD), Momentum, Nesterov Accelerated Gradient (NAG) and RMSProp) from scratch. Loss functions ( Cross Entropy Loss, Mean Squared Error (MSE)). Weight initialization (Random Initialization and Xavier initialization)
- Training neural networks without frameworks
- Hyperparameter tuning using W&B Sweeps
- Experiment tracking and visualization using Weights & Biases

## Project Structure

```
DA6401_ASSIGMENT_01
│
├── README.md
├── requirements.txt
├── src
│   ├── train.py
│   ├── sweep.py
│   ├── inference.py
│   │
│   ├── ann
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   │
│   └── utils
│       ├── __init__.py
│       └── data_loader.py
```

## Installation
- Clone repository: 
''' git clone https://github.com/Abdull811/DA6401_assigment_01.git   '''
- Install dependicies:
''' !pip install -r requirements.txt '''
- Training model:
''' ! python src/train.py \
   -d mnist \
   -e 10 \
   -b 64 \
   -lr 0.001\
   -o rmsprop \
   -nhl 3 \
   -sz 128 64 \
   -a sigmoid \
   -l cross_entropy \
   -wd 0.0001 \
   -wi xavier \
   -wp da6401_Assignment_o1_weight_and_bias

   
