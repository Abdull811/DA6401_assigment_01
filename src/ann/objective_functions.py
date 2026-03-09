"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

class CrossEntropyLoss:
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, y_true, logits):

        probs = self.softmax(logits)
        self.probs = probs

        m = y_true.shape[0]

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        loss = -np.mean(np.log(probs[np.arange(m), y_true] + 1e-9))

        return loss

    def backward(self, y_true, logits):

        if not hasattr(self, "probs"):
            self.probs = self.softmax(logits)

        probs = self.probs.copy()

        m = y_true.shape[0]

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        probs[np.arange(m), y_true] -= 1
        probs /= m

        return probs
    
class MSELoss:
    def forward(self, y_true, logits):
        m = y_true.shape[0]
        y_onehot = np.zeros_like(logits)
        y_onehot[np.arange(m), y_true] = 1

        self.y_onehot = y_onehot
        self.logits = logits

        loss = np.sum((y_onehot - logits) ** 2) / m
        return loss

    def backward(self, y_true, logits):
        m = y_true.shape[0]
        y_onehot = np.zeros_like(logits)
        y_onehot[np.arange(m), y_true] = 1
        return 2 * (logits - y_onehot) / m
