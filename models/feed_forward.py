import torch.nn as nn
from models.silu import SiLU

class FeedForward(nn.Module):
  def __init__(self, embed_dim, intermediate_size):
    super().__init__()

    self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=False)
    self.fc2 = nn.Linear(embed_dim, intermediate_size, bias=False)
    self.swish = SiLU()
    self.fc3 = nn.Linear(intermediate_size, embed_dim, bias=False)

  def forward(self, input):
    ## input:  (batch_size, seq_length, embed_dim)
    ## output: (batch_size, seq_length, embed_dim)

    x1 = self.fc1(input)
    x2 = self.fc2(input)
    output = self.swish(x1) * x2


    return self.fc3(output)