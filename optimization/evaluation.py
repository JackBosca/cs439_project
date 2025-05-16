import torch
from tqdm import tqdm

# def compute_loss(model, dataloader, device):
#     model.eval()
#     total_loss = 0.0
#     total_items = 0
    
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)
#             outputs = model(batch, labels=batch)
#             loss = outputs.loss
#             total_loss += loss.item() * batch.size(0)  # Sum losses
#             total_items += batch.size(0)  # Count items
    
#     # Check empty dataloader
#     if total_items == 0:
#         return float('nan'), float('nan')  
    
#     avg_loss = total_loss / total_items  # Average loss per item
#     perplexity = torch.exp(torch.tensor(avg_loss)).item()  # Perplexity
    
#     return avg_loss, perplexity

@torch.no_grad()
def compute_loss(
    model,
    dataloader,
    device,
    loss_fn=None,  # Optional custom loss function
    return_perplexity=True
):
    model.eval()
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

    if return_perplexity:
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity
    else:
        return avg_loss
