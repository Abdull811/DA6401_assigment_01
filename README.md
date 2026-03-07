# ASSIGMENT 1: Multi-Layer Perceptron For Image Classification

## Overview 
This assigments a Multi-layer perceptron (MLP) neural network from acratch using numpy for image classification. Experiments are tracked using Weight & Biases for visualization and hyperparameter tuning.

## Repository 
- **GitHub Repository:** [https://github.com/Abdull811/DA6401_assigment_01.git]
- **Weights & Biases Run Link:** View Run on W&B [https://wandb.ai/ge26z811-zan/da6401_Assigment_01_weight_bias/table?nw=nwuserge26z811]
- **Weights & Biases Report:** [https://wandb.ai/ge26z811-zan/da6401_Assigment_01_weight_bias/reports/Weight-and-Biases-Report--VmlldzoxNjEzMTk1MQ]

  
## Learniing Objective
- Forward propagation in neural networks
- Backpropagation and gradient computation
- Implementing activations (ReLu, Sigmoid and Tanh) and their derivatives
- Implementing optimization algorithms (Stochastic Gradient Descent (SGD), Momentum, Nesterov Accelerated Gradient (NAG) and RMSProp) from scratch. Loss functions ( Cross Entropy Loss, Mean Squared Error (MSE)). Weight initialization (Random Initialization and Xavier initialization)
- Training neural networks without frameworks
- Hyperparameter tuning using W&B Sweeps
- Experiment tracking and visualization using Weights & Biases

## Project Structure

The project follows a modular structure as shown below:
```
DA6401_ASSIGMENT_01
│
├── README.md
├── requirements.txt
│
├── src
│   ├── train.py                          # Training script for the neural network
│   ├── sweep.py                          # Hyperparameter sweep experiments (100 runs)
│   ├── inference.py                      # Model evaluation and performance metrics
│   │
│   ├── ann                               # Neural network implementation
│   │   ├── __init__.py
│   │   ├── activations.py                # Activation functions (ReLU, Sigmoid, Tanh)
│   │   ├── neural_layer.py               # Implementation of a neural network layer
│   │   ├── neural_network.py             # Complete MLP architecture
│   │   ├── objective_functions.py        # Loss functions (MSE, Cross-Entropy)
│   │   └── optimizers.py                 # Optimizers (SGD, Momentum, NAG, RMSProp)
│   │
│   └── utils                         
│       ├── __init__.py
│       └── data_loader.py                # Dataset loading and preprocessing
```

## Installation
- Clone repository: 
```
 git clone https://github.com/Abdull811/DA6401_assigment_01.git
```

- Install dependicies:
```
!pip install -r requirements.txt
```

- Training model:
```
 ! python src/train.py \
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
```

```
!python -m src.sweep
```
- Model Performance
 ```
  - !python -m src.inference \
       -m src/best_model.npy \
       -d mnist \
       -nhl 2 \
       -sz 128 64 \
       -a relu \
       -l cross_entropy \
       -wi xavier \
       -wd 0.0005 \
       -lr 0.01
    ```

   
