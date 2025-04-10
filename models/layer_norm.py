import torch
import torch.nn as nn

class RMSNorm(nn.Module):
  def __init__(self, embed_dim,  eps = 1e-5):
    super().__init__()
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(embed_dim)) #  (embed_dim)

  def forward(self, input):
    ## input:  (batch_size, seq_length, embed_dim)
    ## output: (batch_size, seq_length, embed_dim)

    input_squared = input ** 2
    RMS = torch.sqrt(input_squared.mean(dim = -1, keepdim = True) + self.eps)
    normalized = input / RMS
    return self.scale * normalized


normal = RMSNorm(1024)
x = torch.rand(10,12,1024)
normal(x).shape