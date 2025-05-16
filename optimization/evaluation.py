import torch
from tqdm import tqdm

@torch.no_grad()
def compute_loss(model, dataloader, device, loss_fn=None, return_perplexity=True):
    """
    Compute the average loss of the model.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        loss_fn: Loss function to use. If None, the model's loss will be used.
        device: Device to perform computations on. ('cuda' or 'cpu')
        return_perplexity: If True, return perplexity as well.
    Returns:
        average_loss: The average loss of the model.
        perplexity: The perplexity of the model (if return_perplexity is True).
    """
    model.eval()

    # Initialize total loss and tokens
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

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
