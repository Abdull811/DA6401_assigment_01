"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

# Cross entropy loss that contain linear combination 
class CrossEntropyLoss:

    def forward(self, y_true, logits):
        """
        y_true: shape (batch_size,) > integer labels
        logits: shape (batch_size, num_classes)
        """
        m = y_true.shape[0]
        # Softmax 
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.probs = probs
        self.y_true = y_true

        correct_probs = probs[np.arange(m), y_true]
        loss = -np.mean(np.log(correct_probs + 1e-9))
        
        return loss

    def backward(self):
        m = self.y_true.shape[0]

        grad = self.probs.copy()
        grad[np.arange(m), self.y_true] -= 1
        grad = grad/m
        return grad

class MSELoss:
    
    def forward(self, y_true, logits):
        m = y_true.shape[0]
        
        # One-hot encode labels
        y_onehot = np.zeros_like(logits)
        y_onehot[np.arange(m), y_true] = 1

        self.y_onehot = y_onehot
        self.logits = logits

        loss = np.sum((y_onehot - logits) ** 2) / m
        return loss

    def backward(self):
        
        m = self.y_onehot.shape[0]
        return 2 * (self.logits - self.y_onehot) / m
