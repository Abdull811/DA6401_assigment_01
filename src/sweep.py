import wandb
from src.train import main

sweep_config = {'method': 'random', 'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {'learning_rate': {'values': [0.0001, 0.001, 0.01]},
    'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
    'activation': {'values': ['relu', 'sigmoid', 'tanh']},
    'hidden_size': {'values': [[64,64], [128,128], [128,64]]}, 'num_layers': {'values': [2,3]},
    'batch_size': {'values': [32,64]}, 'epochs': {'value': 5}, 'dataset': {'value': 'mnist'},
    'loss': {'value': 'cross_entropy'}, 'weight_decay': {'value': 0.0001}, 'weight_init': {'value': 'xavier'}}}

sweep_val = wandb.sweep(sweep_config, project="da6401_Assigment_01_weight_bias")
wandb.agent(sweep_val, function=main, count=100)
print("Sweep ID :", sweep_val)
