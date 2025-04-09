#silu.py

import torch
import torch.nn as nn

class SiLU(nn.Module):
  """ Sigmoid Linear Unit (SiLU) | Swish activation function implementation. """

  def __init__(self):
    """ Initializes the SiLU activation function. """
    super().__init__()

  def forward(self, input):
    return input * torch.sigmoid(input)
silu = SiLU()
x = torch.rand(10,12,1024)
normal(x).shape