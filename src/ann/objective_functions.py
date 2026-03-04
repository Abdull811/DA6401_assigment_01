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
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits_stable)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.probs = probs
        self.y_true = y_true

        correct_class_probs = probs[np.arange(m), y_true]

        loss = -np.mean(np.log(correct_class_probs + 1e-8))
        return loss

    def backward(self):
        m = self.y_true.shape[0]

        grad = self.probs.copy()
        grad[np.arange(m), self.y_true] -= 1
        grad = m

        return grad
class MSELoss:
      def forward(self, y_true, logits):
          self.y_true = y_true
          self.logits = logits
          m = y_true.shape[0]
          return np.sum((y_true - logits) ** 2) / m

      def backward(self):
          m = self.y_true.shape[0]
          return 2 * (self.logits - self.y_true) / m       