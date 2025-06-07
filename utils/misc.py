import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.preprocessing import load_and_preprocess_data
from data.datasets import TextDataset
from models.models_utils import get_tokenizer

# Set seed
def set_seed(seed=42):
    # Python built-ins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_loaders(config, batch_size=4):
    """
    Create data loaders for training, validation, and test datasets.
    Args:
        config: Configuration object containing model and data parameters.
        batch_size (int): Batch size for the data loaders.
    Returns:
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        test_loader: DataLoader for the test dataset.
    """
    # Load data
    train_texts, val_texts, test_texts = load_and_preprocess_data(config)

    # Create datasets
    train_dataset = TextDataset(train_texts, config.tokenizer, config.max_length)
    val_dataset = TextDataset(val_texts, config.tokenizer, config.max_length)
    test_dataset = TextDataset(test_texts, config.tokenizer, config.max_length)
    
    # Create dataloaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader
