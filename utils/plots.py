import matplotlib.pyplot as plt

def plot_loss_curves(results_list):
    plt.figure(figsize=(12, 6))
    for results in results_list:
        label = f"{results['optimizer']}, lr={results['lr']}, bs={results['batch_size']}"
        plt.plot(results['train_loss_history'], label=f"Train - {label}")
        plt.plot(results['val_loss_history'], '--', label=f"Val - {label}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid()
    plt.show()