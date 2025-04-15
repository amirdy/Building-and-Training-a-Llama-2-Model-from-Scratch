from torch.utils.data import DataLoader
from dataset.dataset import Dataset
from transformers import LlamaTokenizer
from torch.utils.data.distributed import DistributedSampler

class DataModule():
    """ Data module for handling training and validation datasets with tokenized inputs. """

    def __init__(self, batch_size, context_length, training_set_tokens, validation_set_tokens):
        """Initializes the DataModule with tokenized datasets.

        Args:
            batch_size (int): Batch size for DataLoader.
            context_length (int): Context length for token sequences.
            training_set_tokens (Any): Tokenized training data.
            validation_set_tokens (Any): Tokenized validation data.
        """
        self.batch_size = batch_size
        self.tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_7b")

        self.train_dataset = Dataset(training_set_tokens, context_length)
        self.val_dataset = Dataset(validation_set_tokens, context_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = False, sampler = DistributedSampler(self.train_dataset, shuffle=True), batch_size = self.batch_size, drop_last = True, num_workers = 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False, drop_last = False, num_workers = 0)