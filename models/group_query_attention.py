import torch
import torch.nn as nn
from torch.nn import functional as F

class GQA(nn.Module):
  """ Multi-Head Attention (MHA) module with optional causal masking. """

  def __init__(self, num_groups, embed_dim, num_heads, context_length, mask = False):
      """ Initializes the Multi-Head Attention module.

      Args:
          embed_dim (int): Dimension of the input embeddings.
          num_heads (int): Number of attention heads.
          drop_rate (float): Dropout rate.
          mask (bool, optional): Whether to apply causal masking. Defaults to False.
      """
      super().__init__()
      self.num_groups = num_groups
      self.group_size = embed_dim // num_groups
      self.head_dim = embed_dim // num_heads
      assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

      self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
      self.W_key = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=False)
      self.W_value = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=False)
      self.out_proj = nn.Linear(embed_dim, embed_dim)  # Linear layer to combine head outputs
      # self.drop = nn.Dropout(drop_rate)
      self.num_heads = num_heads
      self.embed_dim = embed_dim

      self.masked = mask
      self.cos, self.sin = self._precompute_cos_sin(self.head_dim, context_length)


  def _precompute_cos_sin(self, embed_dim, context_length):
    assert embed_dim % 2 == 0, "Embedding Dimension must be even!"
    i_s = torch.arange(1, (embed_dim / 2) + 1)
    theta = 1000 ** (-2 * (i_s - 1) / embed_dim) # [theta_1, theta_2, ..., theta_d/2]  Shape : (d/2)
    theta = torch.tensor([item for item in theta for _ in range(2)]) # [theta_1, theta_1, theta_2, theta_2, ..., theta_d/2, theta_d/2]  Shape : (d)
    theta = theta.unsqueeze(0) # Shape : (1, d)
    m = torch.arange(context_length)  # Shape : (context_length)
    m = m.unsqueeze(1)  # Shape : (context_length, 1)
    # print(m.shape)
    # print(theta.shape)
    angles = m * theta  # Shape : (context_length, d)   [[m1.theta_1, m1.theta_1, m1.theta_2, m1.theta_2, ... ][m2.theta_1, m2.theta_1, m2.theta_2, m1.theta_2, ....][........]]
    # print(angles.shape)
    cos = torch.cos(angles)  # Shape : (context_length, d)
    sin = torch.sin(angles)  # Shape : (context_length, d)
    return cos, sin

  def _rotate(self, x, cos, sin):
    x1 = x[..., ::2]  # even index
    x2 = x[..., 1::2]  # odd index
    x_rotated = torch.stack([-x2, x1], dim=-1)
    x_rotated = x_rotated.flatten(-2)
    print(x.shape, cos.shape, sin.shape, x_rotated.shape)
    return x * (self.cos) + x_rotated * (self.sin)

  def forward(self, input):
      """Computes multi-head attention.

      Args:
          input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

      Returns:
          torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
      """

      batch_size, seq_length, _ = input.shape

      Q = self.W_query(input)  # (batch_size, seq_length, embed_dim)
      K = self.W_key(input)    # (batch_size, seq_length, num_groups * head_dim)
      # K = K.repeat_interleave(self.group_size, dim=-1) # (batch_size, seq_length, num_groups x group_size) # num_groups x group_size = embed_dim
      V = self.W_value(input)  # (batch_size, seq_length, num_groups * head_dim)
      # V = V.repeat_interleave(self.group_size, dim=-1) # (batch_size, seq_length, num_groups x group_size) # num_groups x group_size = embed_dim

      Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
      K = K.view(batch_size, seq_length, self.num_groups, self.head_dim)  # (batch_size, seq_length, num_groups, head_dim)
      V = V.view(batch_size, seq_length, self.num_groups, self.head_dim)  # (batch_size, seq_length, num_groups, head_dim)

      Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
      K = K.permute(0, 2, 1, 3)  # (batch_size, num_groups, seq_length, head_dim)
      V = V.permute(0, 2, 1, 3)  # (batch_size, num_groups, seq_length, head_dim)



      ###       Rotary       ### no change in the shape
      K = self.rotate(K, self.sin, self.cos)
      Q = self.rotate(Q, self.sin, self.cos)

      #########################
      K = K.repeat_interleave(self.group_size, dim=1) # (batch_size, num_groups * group_size, seq_length, head_dim) #  num_groups x group_size = num_heads
      V = V.repeat_interleave(self.group_size, dim=1)  # (batch_size, num_groups * group_size, seq_length, head_dim) #  num_groups x group_size = num_heads



      # Uncomment the following block to use regular attention instead of flash attention
      '''
      d_k = torch.tensor(Q.shape[-1], dtype = torch.float32)
      attn_scores = Q@(K.transpose(2,3))
      # K.transpose(2,3) : (batch_size, num_heads, seq_length, head_dim)
      # attn_scores: : (batch_size, num_heads, seq_length, seq_length)

      if self.masked:
        mask = torch.tril(torch.ones((seq_length, seq_length))).to(attn_scores.device)
        attn_scores = mask * attn_scores
        attn_scores.masked_fill_(attn_scores == 0, float('-inf')) # equal to attn_scores[ attn_scores== 0] = -torch.inf

      attn_weights  = torch.softmax(attn_scores / torch.sqrt(d_k), dim = -1)
      attn_weights = self.drop(attn_weights)
      output = attn_weights@V # (batch_size, num_heads, seq_length, head_dim)
      '''

      # Use flash attention instead of regular attention to speed up
      output = F.scaled_dot_product_attention(Q, K, V, is_causal=self.masked) # flash attention

      output = output.permute(0, 2, 1, 3) # (batch_size, seq_length, num_heads, head_dim)
      output = output.contiguous().view(batch_size, seq_length, self.embed_dim) # (batch_size, seq_length, embed_dim)
      output = self.out_proj(output)  # (batch_size, seq_length, embed_dim)

      return output