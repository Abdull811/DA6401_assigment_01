"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from src.ann.neural_layer import NeuralLayer
from src.ann.activations import ReLu, Sigmoid, Tanh
from src.ann.objective_functions import CrossEntropyLoss, MSELoss

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.layers = []
        self.activations = []

        input_dim = 784
        hidden_sizes = cli_args.hidden_size
        activation_name = cli_args.activation
        weight_init = cli_args.weight_init
        loss_name = cli_args.loss
        self.loss_name = cli_args.loss
        self.weight_decay = cli_args.weight_decay
        self.learning_rate  = cli_args.learning_rate

        layer_dim = [input_dim] + hidden_sizes + [10]
        # Layers
        for i in range(len(layer_dim)-1):
            self.layers.append(NeuralLayer(layer_dim[i], layer_dim[i + 1], weight_init))

            # Activation function for hidden layers
            if i < len(layer_dim) - 2:
               if activation_name == "relu" :
                   self.activations.append(ReLu())
               elif activation_name == "sigmoid":
                   self.activations.append(Sigmoid())
               elif activation_name == "tanh" :
                   self.activations.append(Tanh())

        # Objective function
        if loss_name == "cross_entropy":
           self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MSELoss()

        self.weight_decay = cli_args.weight_decay
        self.learning_rate = cli_args.learning_rate                           

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        x_v = X
        # Hidden layers
        for i in range(len(self.activations)):
            z = self.layers[i].forward(x_v)
            x_v = self.activations[i].forward(z)
            
            # log activation stats for first hidden layer
            if wandb.run is not None:
                wandb.log({"layer1_activation_mean": np.mean(x_v), # logs mean activation 
                           "layer1_activation_fraction": np.mean(x_v == 0)}) # logs fraction of neuron that output =0
        # Last layer 
        logits = self.layers[-1].forward(x_v)
        return logits   

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        self.loss_fn.forward(y_true, y_pred)
        dz = self.loss_fn.backward()

        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse
        # Output layer backpropagation
        da = self.layers[-1].backward(dz, self.weight_decay)
        grad_W_list.append(self.layers[-1].grad_w)
        grad_b_list.append(self.layers[-1].grad_b)
        
        # Hidden layer backpropagation
        for i in reversed(range(len(self.activations))):
            dz = self.activations[i].backward(da)
            da = self.layers[i].backward(dz, self.weight_decay)

            grad_W_list.append(self.layers[i].grad_w)
            grad_b_list.append(self.layers[i].grad_b)
            
        # Log gradient normalization of first hidden layer
        if len(self.layers) > 1:
            grad_norm_layer1 = np.linalg.norm(self.layers[0].grad_w)

            if wandb.run is not None:
               wandb.log({"grad_layer1_norm": grad_norm_layer1})

        # log gradient of 5 neuroins in first layer
        if len(self.layers) > 0:
            first_layer_grad = self.layers[0].grad_w
            for nidx in range(min(5, first_layer_grad.shape[0])): # For 1st 5 neuron
                if wandb.run is not None:
                   wandb.log({f"grad_neuron_{nidx}": np.linalg.norm(first_layer_grad[nidx])})

        # Create explicit object arrays to avoid numpy trying to broadcast shapes
        # For store gradients
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        #print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        #print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b
    
    # weight update
    #def update_weights(self):
     #   for j in self.layers:
      #      j.w -= self.learning_rate * j.grad_w
       #     j.b -= self.learning_rate * j.grad_b
    
    # Training
    def train(self, X_train, y_train, epochs=1, batch_size=32):
        n = X_train.shape[0]
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                logits = self.forward(X_batch)
                self.backward(y_batch, logits)
                self.update_weights()
    # evaluation
    def evaluate(self, X, y):
        logits = self.forward(X)
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y)
        
        return accuracy
    
    # save weights 
    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"w{i}"] = layer.w.copy()
            weights[f"b{i}"] = layer.b.copy()
        return weights
    
    # Load weights
    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"w{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.w = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()