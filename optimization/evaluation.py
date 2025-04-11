import torch

def compute_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item() * batch.size(0)  # Sum losses
            total_items += batch.size(0)  # Count items
    
    # Check empty dataloader
    if total_items == 0:
        return float('nan'), float('nan')  
    
    avg_loss = total_loss / total_items  # Average loss per item
    perplexity = torch.exp(torch.tensor(avg_loss)).item()  # Perplexity
    
    return avg_loss, perplexity