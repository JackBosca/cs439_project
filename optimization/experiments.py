# import torch
import numpy as np
from utils.misc import create_loaders
from models.models_utils import get_model, get_tokenizer
from optimization.training import train_epoch
from optimization.evaluation import compute_loss
from visualization.visual import visual_segment, visual_2D, generate_filter_normalized_vectors
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
    train_loader, val_loader, _ = create_loaders(config, batch_size=batch_size)
    
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

    # Results dictionary
    results = {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'val_perplexity': val_perplexity,
        'optimizer': optimizer_class.__name__,
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'shuffle_mode': shuffle_mode,
        'train_loss_history': train_losses,
        'val_loss_history': val_losses
    }
    
    return results, model

def plot_experiment(model_sb, model_lb, train_loader, val_loader, device, paths: np.array):
    """
    Plot the results of the experiment by visualizing segments and 2D representations of the model parameters.
    Args:
        model_sb: Model with smaller batch size.
        model_lb: Model with larger batch size.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        paths: Array of paths to save the plots.
    """
    # Ensure paths are valid
    assert len(paths)==3
    
    # Visualize segments and 2D representations
    nn = 30
    visual_segment(model_sb, model_lb, nn, compute_loss, train_loader, val_loader, save_path=paths[0], device=device)

    # Generate orthogonal directions for visualization for both models
    a, b = generate_filter_normalized_vectors(model_sb)
    visual_2D(model_sb, a, b, compute_loss, train_loader, n=4, save_path=paths[1])

    c, d = generate_filter_normalized_vectors(model_lb)
    visual_2D(model_lb, c, d, compute_loss, train_loader, n=4, save_path=paths[2])

def calc_sharpness(model, train_loader, device='cpu'):
    """
    Calculate the sharpness of the model using Hessian- and epsilon-based methods.
    Args:
        model: The trained model.
        train_loader: DataLoader for training data.
        device: Device to perform calculations on (e.g., 'cpu' or 'cuda').
    Returns:
        sharpness: Calculated sharpness value.
        hv_norm: Norm of the Hessian-vector product.
    """
    # Compute Hessian-vector product using power iteration
    hv_norm, v = power_iteration_hessian(model, train_loader, device)

    # Compute epsilon-based sharpness
    sharpenss, _ = compute_epsilon_hessian_sharpness(model, train_loader, None, v, device=device)

    # Check sharpness approximation with Taylor expansion
    # Note: The hv_norm is used as the Hessian λ_max in this context
    _ = check_sharpness_approximation(model, train_loader, v, hv_norm, device=device)

    return sharpenss, hv_norm
