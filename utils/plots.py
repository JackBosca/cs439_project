import matplotlib.pyplot as plt
import os

def plot_loss_curves(results_list, save_path=None, fname=None):
    plt.figure(figsize=(12, 6))

    # Use a clean, qualitative colormap (e.g., tab10)
    color_map = plt.get_cmap('tab10')
    num_colors = color_map.N
    
    # Loop through each result and assign color
    for idx, results in enumerate(results_list):
        color = color_map(idx % num_colors)
        label_base = f"{results['optimizer']}, lr={results['lr']}, wd={results['weight_decay']}, bs={results['batch_size']}, order={results['shuffle_mode']}"

        # Plot train and val with same color, different styles
        plt.plot(results['train_loss_history'], label=f"Train - {label_base}", color=color, linestyle='-')
        plt.plot(results['val_loss_history'], label=f"Val - {label_base}", color=color, linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)

    if save_path is not None and fname is not None:
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.join(save_path, f"loss_curves_{fname}.png")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {file_name}")

    plt.show()
    plt.close()
