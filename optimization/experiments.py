from torch.utils.data import DataLoader

from data.datasets import TextDataset
from data.preprocessing import load_and_preprocess_data
from models.models_utils import get_model, get_tokenizer
from optimization.training import train_epoch
from optimization.evaluation import compute_loss

def run_experiment(config, optimizer_class, lr, batch_size, epochs=5, shuffle_mode='random'):
    # Initialize tokenizer
    tokenizer, vocab_size = get_tokenizer(config.model_name)
    config.tokenizer = tokenizer  # Make tokenizer available in config
    config.vocab_size = vocab_size

    # Load data
    train_texts, val_texts, _ = load_and_preprocess_data(config)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, config.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Initialize model
    model = get_model(config.model_name, vocab_size, config.device)
    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device, shuffle_mode)
        val_loss, val_perplexity = compute_loss(model, val_loader, config.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    results = {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_perplexity': val_perplexity,
        'optimizer': optimizer_class.__name__,
        'lr': lr,
        'batch_size': batch_size,
        'shuffle_mode': shuffle_mode,
        'train_loss_history': train_losses,
        'val_loss_history': val_losses
    }
    
    return results