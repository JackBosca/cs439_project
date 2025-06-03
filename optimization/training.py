import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, shuffle_mode='random'):
    """
    Train the model for one epoch.
    Args:
        model: The model to train.
        dataloader: DataLoader for the dataset.
        optimizer: Optimizer for the model.
        device: Device to perform computations on. ('cuda' or 'cpu')
        shuffle_mode: Mode for shuffling data. ('random', 'sorted', or None)
    Returns:
        avg_loss: The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()

    # Initialize total loss and total tokens
    total_loss = 0.0
    total_tokens = 0

    if shuffle_mode == 'random':
        dataloader.dataset.shuffle()
    elif shuffle_mode == 'sorted':
        dataloader.dataset.sort_by_length()

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Batch is input_ids (or (input_ids, attention_mask))
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None
        else:
            inputs = batch.to(device)
            attention_mask = None

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=str(device), enabled=True, dtype=torch.float16):
            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)

        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Multiply by number of tokens
        total_loss += loss.item() * inputs.numel()

        # Count tokens
        total_tokens += inputs.numel()

    # Compute average loss as total loss divided by total tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    return avg_loss
