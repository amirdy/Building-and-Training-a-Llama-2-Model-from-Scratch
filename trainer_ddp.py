import torch
import torch.nn as nn
import time
import numpy as np
import math
from pathlib import Path
from torch.distributed import  destroy_process_group

class Trainer:
    """ Trainer class for training and evaluating a Llama2 model. """
    def __init__(self, tokenizer, train_dataloader, val_dataloader, model, config,  device, sample_context, ddp_world_size, master_process, ddp):
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.iter_train_dataloader = iter(train_dataloader)
        self.val_dataloader = val_dataloader
        self.model = model
        self.config = config
        self.device = device
        self.device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'

        self.ddp_world_size = ddp_world_size
        self.master_process = master_process
        self.ddp = ddp
        self.sample_context = sample_context
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._configure_optimizer()
        self.grad_accum_steps = config.grad_accum_steps // ddp_world_size # Adjust the gradient accumulation steps for distributed training
        
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.train_losses = []
        self.val_losses = []
        self.checkpoint_dir = Path("ckpt")
        self._prepare_checkpoint_dir()

    def _prepare_checkpoint_dir(self):
        """ Prepare the checkpoint directory. """
        if self.master_process:
            try:
                self.checkpoint_dir.mkdir(parents=True)
                print(f"Checkpoint directory is ready at: {self.checkpoint_dir}...")
            except FileExistsError:
                print(f"The checkpoint directory ({self.checkpoint_dir}) already exists...")

    def _get_batch(self):
        """ Get a batch of data from the training dataloader. """
        try:
            return next(self.iter_train_dataloader)
        except StopIteration: # If the iterator is exhausted, create a new one
            self.iter_train_dataloader = iter(self.train_dataloader)
            return next(self.iter_train_dataloader)

    def _train_step(self, current_step):
        """ Perform a single training step. """
        self.model.train()
        # loss_accum = 0
        loss_accum = torch.zeros(1, device=self.device)

        # Zero the gradients
        self.optimizer.zero_grad()

        for i in range(self.grad_accum_steps):
            X, y = self._get_batch() # X: (batch_size, seq_length)   y: (batch_size, seq_length)
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            with torch.autocast(device_type= self.device_type, dtype=torch.bfloat16): # Use bfloat16 for faster computation
                pred = self.model(X)     # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1)     # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1)           # (batch_size x seq_length)

            # Calculate loss
            loss = self.criterion(pred, y) / self.grad_accum_steps
            loss_accum += loss.detach()
            if self.ddp:
                self.model.require_backward_grad_sync = (i == self.grad_accum_steps) # Synchronize gradients across all processes
            loss.backward()
        
        if self.ddp:
            # loss_accum_ = torch.tensor(loss_accum, dtype=torch.float32, device=self.device) # Create a tensor to hold the loss for all processes
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG) # Average the loss across all processes
        
        
        # Clip norm of gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update learning rate
        lr = self._get_lr(current_step)
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Update weights
        self.optimizer.step()

        return loss_accum, lr


    def _evaluate(self):
        """ Evaluate the model on the validation set. """
        self.model.eval()
        losses= []
        for X, y in self.val_dataloader:
            X, y = X.to(self.device), y.to(self.device) # X: (bath_size, seq_length)   y: (bath_size, seq_length)

            # Forward pass
            with torch.no_grad(): # No need to track the gradients
                with torch.autocast(device_type= self.device_type, dtype=torch.bfloat16): # Use bfloat16 for faster computation
                    pred = self.model(X) # (batch_size, seq_length, vocab_size)
            pred = pred.flatten(0,1)     # (batch_size x seq_length, vocab_size)
            y = y.flatten(0,1)           # (batch_size x seq_length)
            loss = self.criterion(pred, y)
            losses.append(loss.item())

        val_loss = np.mean(losses)
        return val_loss


    def _log_results(self, step, train_loss, val_loss, training_time, lr):
        """ Log the training results. """
        print(f'Device: {self.device} | Step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f} | lr: {lr:.4f} | {training_time:.2f}s ')

    def _save_checkpoint(self, step, val_loss):
        """ Save the model checkpoint if the validation loss has improved. """
        
        if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_step = step
                    model_filename = self.checkpoint_dir / f'best_model.pth'
                    torch.save(self.model.state_dict(), model_filename)

    def _print_sample_output(self):
        """ Generate and print a sample output from the model. """
        tokens = self.tokenizer.encode(self.sample_context) # list of indexes [3, 2, 1, ... ]
        for i in range(self.config.max_new_token):
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Generate logits from the model for the current token sequence
                logits = self.model(tokens_tensor)
                # Extract logits corresponding to the last token in the sequence (shape: [vocab_size])
                last_seq_logits = logits[0, -1, :]
                # Scale logits using temperature
                scaled_logits = last_seq_logits / self.config.temperature
                # Convert scaled logits to probabilities using softmax
                probs = torch.softmax(scaled_logits, dim=0)
                # Sort the probabilities in descending order and get sorted indices
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                # Create a mask for tokens to keep — cumulative probability <= top_p
                nucleus_mask = cumulative_probs <= self.config.top_p
                # Ensure at least one token is included
                nucleus_mask[0] = True
                # Get the final indices to sample from
                nucleus_indices = sorted_indices[nucleus_mask]
                # Create a new distribution only over the nucleus
                nucleus_probs = probs[nucleus_indices]
                nucleus_probs = nucleus_probs / nucleus_probs.sum()  # Normalize
                # Sample the next token from the nucleus
                next_token = nucleus_indices[torch.multinomial(nucleus_probs, num_samples=1)]
                # Append the sampled token to the sequence
                tokens = tokens + [next_token.item()]

        decoded_text = self.tokenizer.decode(tokens) .replace("\n", " ")
        print(f'> {decoded_text}')

    def _configure_optimizer(self):
        """ Configure the optimizer. """
        # Get all trainable parameters
        param_dict = {name: p for name, p in self.model.named_parameters() if  p.requires_grad}
        # Apply weight decay to all weights except biases and batch norm layers
        param_groups = [
            {'params': [p for name, p in param_dict.items() if p.dim() >= 2 ], 'weight_decay': self.config.weight_decay},
            {'params': [p for name, p in param_dict.items() if p.dim() < 2 or not p.requires_grad], 'weight_decay': 0.0}
        ]

        optmizer = torch.optim.AdamW(param_groups, lr = self.config.max_lr, betas=(0.9, 0.95), eps=1e-5)
        return optmizer

    def _get_lr(self, current_step):
        """ Get the learning rate for the current step based on the learning rate schedule. """
        # (warup) if the step < warmup_steps then increase the learning rate linearly
        if current_step < self.config.warmup_steps:
            return   ((current_step + 1) / self.config.warmup_steps) * self.config.max_lr
        # (constant min LR) if the step > max_steps then decrease the learning rate linearly
        elif current_step > self.config.max_steps:
            return self.config.min_lr
        # (cosine decay) if the step is between warmup_steps and max_steps then decrease the learning rate using cosine decay
        decay_ratio = (current_step - self.config.warmup_steps)  / (self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

        return self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)

    def train(self):
        """ Train the model. """
        for step in range(self.config.max_steps):
            start_time = time.time()
            train_loss, lr = self._train_step(step)
            if step % 100 == 0: # Log the results every 10 steps
                val_loss = self._evaluate()
                end_time = time.time()
                training_time = (end_time - start_time)
                self._log_results(step, train_loss, val_loss, training_time, lr)
                if self.master_process: 
                    self._save_checkpoint(step, val_loss)
                    self._print_sample_output()
        if self.ddp:
            destroy_process_group() # Destroy the process group for distributed training