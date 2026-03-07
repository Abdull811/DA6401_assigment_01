
import numpy as np
import argparse
from sklearn.metrics import f1_score
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ann.neural_network import NeuralNetwork

# Replace these values with your best config if needed
best_config = argparse.Namespace(
    dataset="mnist",
    epochs=2,
    batch_size=64,
    loss="cross_entropy",
    optimizer="sgd",
    weight_decay=0.0,
    learning_rate=0.01,
    num_layers=2,
    hidden_size=[64, 64],
    activation="relu",
    weight_init="xavier"
)

# Initialize model
model = NeuralNetwork(best_config)

# Load saved weights
weights = np.load("src/best_model.npy", allow_pickle=True).item()
model.set_weights(weights)

X_test = np.random.rand(100, 784)  # 100 samples, 784 features
y_true = np.random.randint(0, 10, size=(100,))  # 100 labels in range 0–9

# Forward pass
logits = model.forward(X_test)
y_pred_labels = np.argmax(logits, axis=1)

# Compute F1 score
print("F1 Score:", f1_score(y_true, y_pred_labels, average='macro'))