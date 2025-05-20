import torch
from tqdm import tqdm

@torch.no_grad()
def compute_loss(model, dataloader, device, loss_fn=None, return_perplexity=True, max_batches=None):
    """
    Compute the average loss of the model.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        loss_fn: Loss function to use. If None, the model's loss will be used.
        device: Device to perform computations on. ('cuda' or 'cpu')
        return_perplexity: If True, return perplexity as well.
        max_batches: Maximum number of batches to evaluate. If None, evaluate all batches.
    Returns:
        average_loss: The average loss of the model.
        perplexity: The perplexity of the model (if return_perplexity is True).
    """
    model.eval()

    # Initialize total loss and tokens
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
        # Check if max_batches is reached
        if max_batches is not None and i >= max_batches:
            break

        # Unpack batch
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None
        else:
            inputs = batch.to(device)
            attention_mask = None

        # Forward pass
        outputs = model(inputs, attention_mask=attention_mask, labels=inputs)

        if loss_fn is not None:
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), inputs.view(-1))
        else:
            loss = outputs.loss

        # Accumulate total loss and tokens
        if attention_mask is not None:
            num_tokens = attention_mask.sum().item()
        else:
            num_tokens = inputs.numel()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    # Avoid division by zero
    if total_tokens == 0:
        return float('nan'), float('nan') if return_perplexity else float('nan')

    avg_loss = total_loss / total_tokens

    # Calculate perplexity if required
    if return_perplexity:
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity
    else:
        return avg_loss
