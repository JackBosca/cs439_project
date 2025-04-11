from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, shuffle_mode='random'):
    model.train()
    total_loss = 0
    
    if shuffle_mode == 'random':
        dataloader.dataset.shuffle()
    elif shuffle_mode == 'sorted':
        dataloader.dataset.sort_by_length()
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch, labels=batch)
        
        # Compute loss and backpropagation
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)