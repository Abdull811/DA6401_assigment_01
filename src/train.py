"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork
from src.ann.optimizers import SGD, Momentum, NAG, RMSProp

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", default="mnist", choices=["mnist","fashion_mnist"])
    parser.add_argument("-e","--epochs", default=10, type=int)
    parser.add_argument("-b","--batch_size", default=64, type=int)
    parser.add_argument("-lr","--learning_rate", default=0.01, type=float)
    parser.add_argument("-o","--optimizer", default="sgd", choices=["sgd","momentum","nag","rmsprop"])
    parser.add_argument("-nhl","--num_layers", default=2, type=int)
    parser.add_argument("-sz","--hidden_size", nargs="+", default=[128,64], type=int)
    parser.add_argument("-a","--activation", default="relu", choices=["relu","sigmoid","tanh"])
    parser.add_argument("-l","--loss", default="cross_entropy", choices=["cross_entropy","mse"])
    parser.add_argument("-wd","--weight_decay", default=0.0001, type=float)
    parser.add_argument("-wi","--weight_init", default="xavier", choices=["random","xavier"])
    parser.add_argument("-wp","--wandb_project", default="da6401_Assigment_01_weight_bias")
    parser.add_argument("--model_save_path", default="src/best_model.npy")
    return parser.parse_args()
       
def main(args=None):
    if args is None:
        args = parse_arguments()

    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must equal number of hidden_size values")
    
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config

    # Load Dataset
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

    # Model initialization
    model = NeuralNetwork(args)

    # W&B sample logging (keep this)
    table = wandb.Table(columns=["Image", "Label"])
    y_labels = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    for v in range(10):
        ind = np.where(y_labels == v)[0]
        random_ind = np.random.choice(ind, 5, replace=False)
        for q in random_ind:
            img = x_train[q].reshape(28, 28)
            table.add_data(wandb.Image(img), v)
    wandb.log({f"{args.dataset} Samples": table})
    
    # Initialize optimizer
    if args.optimizer == "sgd":
        optimizer = SGD(args.learning_rate, args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = Momentum(args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        optimizer = NAG(args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(args.learning_rate, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0 

        for i in range(0, x_train_shuffled.shape[0], args.batch_size):
            x_batch = x_train_shuffled[i:i + args.batch_size]
            y_batch = y_train_shuffled[i:i + args.batch_size]

            logits = model.forward(x_batch)
            loss = model.loss_fn.forward(y_batch, logits)
            epoch_loss += loss
            num_batches += 1

            model.backward(y_batch, logits)
            optimizer.update(model.layers)
        
        if num_batches > 0:
            epoch_loss /= num_batches
            train_losses.append(epoch_loss)

        train_logits = model.forward(x_train)
        train_pred = np.argmax(train_logits, axis=1)
        train_acc = np.mean(train_pred == y_train)

        val_logits = model.forward(x_val)
        val_pred = np.argmax(val_logits, axis=1)
        val_acc = np.mean(y_val == val_pred)
        val_accuracies.append(val_acc)
        
        wandb.log({"epoch": epoch, "train_loss": epoch_loss,
                   "train_accuracy": train_acc, "val_accuracy": val_acc})
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.save(args.model_save_path, model.get_weights())
    
    # Training loss curve (keep plots)
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    wandb.log({"Training Loss Curve": wandb.Image(plt.gcf())})
    plt.close()

    plt.figure()
    plt.plot(val_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    wandb.log({"Validation Accuracy Curve": wandb.Image(plt.gcf())})
    plt.close()

    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    best_weights = np.load(args.model_save_path, allow_pickle=True).item()
    model.set_weights(best_weights)

    test_logits = model.forward(x_test)
    test_pred = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_pred == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})

    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(cm, cmap="Blues")

    classes = [str(i) for i in range(10)]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=12)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.colorbar(im) 
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig) 
    wandb.finish()      

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
