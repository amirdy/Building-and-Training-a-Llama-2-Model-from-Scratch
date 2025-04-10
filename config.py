from dataclasses import dataclass

@dataclass
class LlamaConfig_7B:
    """
    Configuration for Llama2_7B model architecture.

    Attributes:
        vocab_size: Vocabulary size used in the model.
        context_length: The maximum context window length.
        emb_dim: The embedding dimension size.
        n_heads: The number of attention heads in the multi-head attention layer.
        n_layers: The number of transformer layers.
        qkv_bias: Whether to use bias in the QKV matrices.
        num_groups: Number of groups for the GQA (Group Query Attention) mechanism.
        intermediate_size: Size of the hidden layer in the feedforward network.
    """
    vocab_size = 32000     ### See the Llama 2 paper
    context_length = 4096  ### See the Llama 2 paper
    emb_dim = 4096         ### See the Llama 1 paper
    n_heads = 32           ### See the Llama 1 paper
    n_layers = 32//2       ### See the Llama 1 paper
    qkv_bias = False
    num_groups = 32         ### 32 = In order to be like Multi-head Attention
    intermediate_size = 11008   ### Size of the hidden in FeedForward


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.

    Attributes:
        max_steps: Total number of training steps.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        max_lr: Maximum learning rate for the optimizer.
        min_lr: Minimum learning rate during training.
        weight_decay: Weight decay coefficient for optimizer.
        batch_size: Batch size for each training step.
        grad_accum_steps: Number of gradient accumulation steps.
        max_new_token: The number of new tokens to generate at each step.
        temperature: Sampling temperature for text generation.
        top_p: Probability threshold for nucleus sampling.
    """
    max_steps = 500  # 2T tokens | batch_size = 4M | so the number of steps should be: 2T/4M = 500,000 steps in Llama2
    warmup_steps = 2 # 2000 in Llama2
    max_lr = 3e-4    # Llama2_7B
    min_lr = 3e-5    # Llama2_7B    (0.1 * 3e-4)
    weight_decay = 0.1   # Llama2_7B
    batch_size = 64  # Should be 4M = 2^22 | context_length=2^12   x   batch_size=2^6   x   grad_accum_steps=2^4
    grad_accum_steps = 16 # See the Batch Size
    max_new_token = 100
    temperature = 1
    top_p = 0.9


