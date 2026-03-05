"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix 
from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork
from src.ann.optimizers import SGD, Momentum, NAG, RMSProp

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument("-d", "--dataset", required=True,
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-o", "--optimizer", required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True)
    parser.add_argument("-a", "--activation", required=True,
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", required=True,
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wd", "--weight_decay", type=float, required=True)
    parser.add_argument("-wi", "--weight_init", required=True,
                        choices=["random", "xavier"])
    parser.add_argument("-wp", "--wandb_project", required=False)
    parser.add_argument("--model_save_path", default="src/best_model.npy")
    
    return parser.parse_args()

def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    wandb.init(project="da6401_Assigment_01_weight_bias",
               config=vars(args))
    config = wandb.config
    args.batch_size = config.batch_size
    args.learning_rate = config.learning_rate
    args.optimizer = config.optimizer
    args.activation = config.activation
    args.hidden_size = config.hidden_size
    args.weight_decay = config.weight_decay
    args.weight_init = config.weight_init
    args.num_layers = config.num_layers
    args.loss = config.loss
    args.epochs = config.epochs
    args.dataset = config.dataset
    
    # Load Dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

    # Model initialization
    model = NeuralNetwork(args)

    table = wandb.Table(columns=["Image", "Label"])

    for v in range(10):
        ind = np.where(y_train == v)[0][:5]
        for q in ind:
            img = x_train[q].reshape(28, 28)
            table.add_data(wandb.Image(img), v)
    
    wandb.log({"MNIST Samples":table})
    
    # Initialize SGD, Momentum, NAG, RMSProp Optimizers
    if args.optimizer == "sgd":
        optimizer = SGD(args.learning_rate)

    elif args.optimizer == "momentum":
        optimizer = Momentum(args.learning_rate)

    elif args.optimizer == "nag":
        optimizer = NAG(args.learning_rate)

    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(args.learning_rate)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        # Random training data
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0
        num_batches = 0 

        for i in range(0, x_train.shape[0], args.batch_size):

            x_batch = x_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            # Forward pass
            logits = model.forward(x_batch)

            # Compute loss
            loss = model.loss_fn.forward(y_batch, logits)
            epoch_loss += loss
            num_batches+= 1

            # Backward pass
            model.backward(y_batch, logits)
            # Update weights
            optimizer.update(model.layers)
        
        if num_batches > 0:
           epoch_loss /= num_batches
        # Validation Accuracy
        val_logits = model.forward(x_val)
        val_pred = np.argmax(val_logits, axis=1)

        val_acc = np.mean(val_pred == y_val)
        
        wandb.log({"epoch":epoch, "train_loss":epoch_loss,
                   "val_accuracy": val_acc})
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Val Accuracy: {val_acc:.4f}")

        # Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights = model.get_weights()
            np.save(args.model_save_path, weights)

    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # load best model
    best_weights = np.load(args.model_save_path, allow_pickle=True).item()
    model.set_weights(best_weights)

    # test evaluation
    test_logits = model.forward(x_test)
    test_pred = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_pred == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})

    # log confusion matrix
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=test_pred)})

if __name__ == '__main__':
    main()