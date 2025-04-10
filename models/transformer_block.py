import torch.nn as nn
from models.layer_norm import RMSNorm
from models.group_query_attention import GQA
from models.feed_forward import FeedForward

class TransformerBlock(nn.Module):
  """A single Transformer block consisting of multi-head attention, normalization, and feed-forward layers."""

  def __init__(self, config):
    """Initializes the Transformer block.

    Args:
        config: A configuration object containing model hyperparameters.
    """
    super().__init__()

    self.GQA = GQA(config.num_groups, config.emb_dim, config.n_heads, config.context_length, mask = True)
    self.layer_nomalization_1 = RMSNorm(config.emb_dim)
    self.layer_nomalization_2 = RMSNorm(config.emb_dim)
    self.feed_forward = FeedForward(config.emb_dim, config.intermediate_size)

  def forward(self, input):
    """Performs the forward pass of the Transformer block.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

    Returns:
          torch.Tensor: Output tensor of the same shape as input.
    """

    shortcut = input
    output = self.layer_nomalization_1(shortcut) # (batch_size, seq_length, embed_dim)
    output = self.GQA(output)                    # (batch_size, seq_length, embed_dim)
    output = output + shortcut

    shortcut = output
    output = self.layer_nomalization_2(shortcut) # (batch_size, seq_length, embed_dim)
    output = self.feed_forward(output)           # (batch_size, seq_length, embed_dim)
    output = output + shortcut

    return output