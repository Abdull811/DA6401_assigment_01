"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

# Cross entropy loss that contain linear combination 
class CrossEntropyLoss:
    def forward(self, y_true, logits):
        # Apply softmax to logits
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        m = y_true.shape[0]
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        # Compute loss
        loss = -np.mean(np.log(probs[np.arange(m), y_true] + 1e-9))
        self.probs = probs
        return loss

    def backward(self, y_true, logits):
        m = y_true.shape[0]
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        grad = self.probs.copy()
        grad[np.arange(m), y_true] -= 1
        grad /= m
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
