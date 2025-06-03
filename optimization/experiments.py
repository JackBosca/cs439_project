import torch
from torch.utils.data import DataLoader

from data.datasets import TextDataset
from data.preprocessing import load_and_preprocess_data
from models.models_utils import get_model, get_tokenizer
from optimization.training import train_epoch
from optimization.evaluation import compute_loss
from sharpness.sharpness import compute_epsilon_hessian_sharpness, power_iteration_hessian, check_sharpness_approximation
 
def run_experiment(config, optimizer_class, lr, weight_decay, batch_size, epochs=5, rand_dir=True, shuffle_mode='random'):
    """
    Run a training experiment with the specified configuration and optimizer.
    Args:
        config: Configuration object containing model and training parameters.
        optimizer_class: Optimizer class to use (e.g., torch.optim.Adam).
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        batch_size: Batch size for training and validation.
        epochs: Number of epochs to train the model.
        rand_dir: Whether to use random direction for sharpness estimation.
        shuffle_mode: Mode for shuffling data ('random' or 'sequential').
    Returns:
        results: Dictionary containing training and validation losses, perplexity, optimizer details, and sharpness metrics.
    """
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
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device, shuffle_mode)
        val_loss, val_perplexity = compute_loss(model, val_loader, config.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    # Sharpness analysis
    print("Estimating Hessian sharpness...")

    # Use cross entropy as loss_fn
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    lambda_max, eigenvector = power_iteration_hessian(
        model, train_loader, device=config.device, num_iters=20, num_batches=3
    )

    base_loss = val_loss
    sharpness, _ = compute_epsilon_hessian_sharpness(
        model, train_loader, loss_fn, eigenvector, epsilon=1e-3, num_samples=3, 
        rand_dir=rand_dir, base_loss=base_loss, device=config.device
    )

    print(f"Hessian λ_max: {lambda_max:.4f}")
    print(f"Sharpness: {sharpness:.2f}%")

    # Check sharpness approximation
    _ = check_sharpness_approximation(
        model, train_loader, eigenvector, lambda_max, epsilon=1e-3, device=config.device
    )

    # Results dictionary
    results = {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_perplexity': val_perplexity,
        'optimizer': optimizer_class.__name__,
        'lr': lr,
        'batch_size': batch_size,
        'shuffle_mode': shuffle_mode,
        'train_loss_history': train_losses,
        'val_loss_history': val_losses,
        'hessian_lambda_max': lambda_max,
        'sharpness': sharpness,
        'sharpness_base_loss': base_loss
    }

    return results