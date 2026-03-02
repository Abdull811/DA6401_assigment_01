"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

# Cross entropy loss that contain linear combination 
class CrossEntropyLoss:
      def forward(self, y_true, logits): 
          self.y_true = y_true
          m = y_true.shape[0]
          
          logits_shift = logits - np.max(logits, axis=1, keepdims=True)
          exp_z = np.exp(logits_shift) 
          self.y_prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
          loss = -np.sum(y_true * np.log(self.y_prob + 1e-8)) / m
          return loss
      
      def backward(self):
          # dl_dz = (y_pro - y_true) / m
          m = self.y_true.shape[0]
          return (self.y_prob - self.y_true) / m
      
class MSELoss:
      def forward(self, y_true, y_pred):
          self.y_true = y_true
          self.y_pred = y_pred
          m = y_true.shape[0]
          return np.sum((y_true - y_pred) ** 2) / m

      def backward(self):
          m = self.y_true.shape[0]
          return 2 * (self.y_pred - self.y_true) / m       