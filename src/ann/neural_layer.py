import numpy as np

# Perform liner z = wx + b and store gradient during back propagation
class NeuralLayer:
      # Weight and bias initialization
      def __init__(self, input_dim, output_dim, weight_init):
          std = np.sqrt(2.0 / (input_dim + output_dim)) # std for xavier
          if weight_init == "random":
             self.w = np.random.randn(input_dim, output_dim)
          elif weight_init ==  "xavier":
               self.w = np.random.randn(input_dim, output_dim) * std
          else: 
               print("Error")
          
          self.b = np.zeros(1, output_dim)
       
    # Forward pass of the layer
      def forward(self, x):
           self.x = x 
           return x @ self.w + self.b # Linear equation(z = xw + b)
      
      # Backward pass of the layer
      def backward(self, dz, weight_decay):
          m = self.x.shape[0] # batch size
          # gradient of weight and bias
          self.grad_w = (self.x.T @ dz) / m + (weight_decay / m) * self.w 
          self.grad_b = np.sum(dz, axis=0, keepdims=True) / m
          return dz @ self.w.T
