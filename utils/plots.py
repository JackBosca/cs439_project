import os
import matplotlib.pyplot as plt

def plot_loss_curves(results_list, save_path=None, fname=None):
    plt.figure(figsize=(12, 6))

    for results in results_list:
        label = f"{results['optimizer']}, lr={results['lr']}, wd={results['weight_decay']}, bs={results['batch_size']}"
        plt.plot(results['train_loss_history'], label=f"Train - {label}")
        plt.plot(results['val_loss_history'], '--', label=f"Val - {label}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid()

    if save_path is not None and fname is not None:
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.join(save_path, f"loss_curves_{fname}.png")
        plt.savefig(file_name)
        print(f"Plot saved to: {file_name}")

    plt.show()
    plt.close()