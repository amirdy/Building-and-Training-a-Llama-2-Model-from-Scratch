import torch
from models.llama import Llama
from dataset.data_module import DataModule
from config import LlamaConfig_7B, TrainingConfig
from trainer import Trainer
import time
import os 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# Load the tiny stories dataset 
def load_tiny_stories():
    tokens_train = []
    tokens_val = []

    with open("train_tokens.txt", "r", encoding="utf-8") as f:
        for line in f:
            # Convert each space-separated token back into an integer list and append to loaded_tokens
            token_list = list(map(int, line.strip().split()))
            tokens_train.extend(token_list)

    with open("val_tokens.txt", "r", encoding="utf-8") as f:
        for line in f:
            # Convert each space-separated token back into an integer list and append to loaded_tokens
            token_list = list(map(int, line.strip().split()))
            tokens_val.extend(token_list)
    return tokens_train, tokens_val



def main():
    """ Main function to set up and train the Llama2-7B model. """
    
    # Load dataset
    tokens_train, tokens_val = load_tiny_stories()



    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend ='nccl')
        ddp_rank = int(os.environ(['RANK']))
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # Only the master process will print logs and save checkpoints
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True

    # Set the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}...')

    # Initialize configurations
    Llama_config = LlamaConfig_7B()
    training_config = TrainingConfig()


    # Initialize Data Module and Data Loaders
    dm = DataModule(training_config.batch_size, Llama_config.context_length, tokens_train, tokens_val)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    # Enable TF32 for faster training on Ampere GPUs
    torch.set_float32_matmul_precision('high') 

    # Create the Llama model
    model = Llama(Llama_config)
    model.to(device) # Move the model to the device

    # Compile the model for faster training
    model = torch.compile(model) 
    if ddp: 
        model = DDP(model, device_ids = [ddp_local_rank]) # Wrap the model with DDP for distributed training
        model = model.module # Get the original model from DDP wrapper

    # Set the sample context
    sample_context = "Once, a cat sees a dog and says, 'Hi puppy! How's life going?' "

    # Create the trainer
    trainer = Trainer(
        tokenizer = dm.tokenizer,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        model = model,
        config = training_config,
        device = device,
        sample_context = sample_context,
        ddp_world_size = ddp_world_size,
        master_process = master_process,
        ddp = ddp
    )
    print('Start training')
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time_minutes = (end_time - start_time)/60
    print(f'\n Training completed in {training_time_minutes:.2f} minutes.')

if __name__ == "__main__":
    main()