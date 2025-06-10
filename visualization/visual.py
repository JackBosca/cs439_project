import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import os

def ensure_plot_dir(dir_path="./exp_plots"):
    """
    Ensure that the directory for saving plots exists.
    If it does not exist, create it.
    Args:
        dir_path (str): The directory path where plots will be saved.
    Returns:
        str: The directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def flatten_params(model):
    """
    Flattens the parameters of a PyTorch model into a single 1D tensor.
    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be flattened.
    Returns:
        torch.Tensor: A 1D tensor containing all the parameters of the model.
    """
    with torch.no_grad():
        return torch.cat([param.data.view(-1) for param in model.parameters()])

def set_flat_params(model, flat_params):
    """
    Sets the parameters of a PyTorch model from a flattened 1D tensor.
    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be set.
        flat_params (torch.Tensor): A 1D tensor containing the parameters to set in the model.
    """
    idx = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.data.numel()
            param.data.copy_(flat_params[idx:idx + numel].view_as(param.data))
            idx += numel

def generate_orthogonal_directions(model):
    """
    Generates two orthogonal directions in the parameter space of the model.
    Args:
        model (torch.nn.Module): The PyTorch model from which to generate directions.
    Returns:
        tuple: Two orthogonal tensors (a, b) representing directions in the parameter space.
    """
    theta = flatten_params(model)
    dim = theta.shape[0]

    a = torch.randn(dim)
    b = torch.randn(dim)

    b = b - torch.dot(a, b) / torch.dot(a, a) * a

    a = a / torch.norm(a)
    b = b / torch.norm(b)

    return a, b 
 
def visual_segment(init_model, final_model, n, loss_fn, train_loader, val_loader, save_path=None, device='cpu'):
    """
    Computes and plots the loss along the segment between two models.
    If save_path is provided, saves the plot (otherwise defaults to 'plots/segment_loss.png').
    Args:
        init_model (torch.nn.Module): The initial model.
        final_model (torch.nn.Module): The final model.
        n (int): Number of interpolation points.
        loss_fn (function): Loss function to compute the loss.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        save_path (str, optional): Path to save the plot. Defaults to None.
        device (str, optional): Device to run the computations on. Defaults to 'cpu'.
    """
    plt.figure()

    # Create alphas for interpolation
    alphas = np.linspace(-0.5, 1.5, n + 1)
    losses_val = []
    losses_train = []
 
    # Flatten the parameters of the initial and final models
    theta_0 = flatten_params(init_model)
    theta_f = flatten_params(final_model)
    currModel = copy.deepcopy(init_model)
    
    for alpha in alphas:
        # Interpolate parameters
        theta = theta_0 + alpha * (theta_f - theta_0)
        set_flat_params(currModel, theta)
        currModel.eval()

        # Compute losses for the current model
        loss_val, _ = loss_fn(currModel, val_loader, device)
        loss_train, _ = loss_fn(currModel, train_loader, device)
        losses_val.append(loss_val)
        losses_train.append(loss_train)

    # Plotting
    plt.plot(alphas, losses_train, label='Train Loss', linestyle='-', color='#1b9e77')
    plt.plot(alphas, losses_val, label='Validation Loss', linestyle='--', color='#d95f02')
    plt.xlabel('Interpolation Factor')
    plt.ylabel('Loss')
    plt.title(r'Loss Along Interpolation Between $\theta_0$ and $\theta_f$')
    plt.legend()
    plt.grid(True)

    # Save the plot
    if save_path is None:
        # Call ensure_plot_dir to create the directory if it doesn't exist
        PLOT_DIR = ensure_plot_dir()
        save_path = os.path.join(PLOT_DIR, 'segment_loss.png')
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close() 

def visual_2D(model, a, b, loss_fn, train_loader, n=50, range_val=1.0, save_path=None):
    """
    Computes and plots the loss surface in the 2D plane defined by two directions a and b.
    If save_path is provided, saves the plot (otherwise defaults to 'exp_plots/loss_surface_2d.png').
    Args:
        model (torch.nn.Module): The PyTorch model.
        a (torch.Tensor): Direction vector a.
        b (torch.Tensor): Direction vector b.
        loss_fn (function): Loss function to compute the loss.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        n (int, optional): Number of points in each direction. Defaults to 50.
        range_val (float, optional): Range for the grid. Defaults to 1.0.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    # Generate a grid of points in the 2D plane defined by a and b
    alphas = np.linspace(-range_val, range_val, n)
    betas = np.linspace(-range_val, range_val, n)
    loss_grid = np.zeros((n, n))

    device = next(model.parameters()).device

    # Ensure a, b are on same device as model
    a = a.to(device)
    b = b.to(device)

    # Flatten the parameters of the model and create a copy
    theta_0 = flatten_params(model).to(device)
    currModel = copy.deepcopy(model).to(device)

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Compute the parameters for the current point in the grid
                theta = theta_0 + alpha * a + beta * b
                set_flat_params(currModel, theta)
                currModel.eval()

                # Compute the loss for the current model
                loss, _ = loss_fn(currModel, train_loader, device)
                loss_grid[i, j] = loss

    A, B = np.meshgrid(alphas, betas)
    plt.figure(figsize=(8, 6))

    # Filled contour plot
    cp = plt.contourf(A, B, loss_grid.T, levels=50)
    # Add contour lines
    lines = plt.contour(A, B, loss_grid.T, levels=20, colors='black', linewidths=0.5)
    plt.clabel(lines, inline=True, fontsize=8)
    cbar = plt.colorbar(cp)
    cbar.set_label('Loss')
    plt.xlabel(r'Direction $\alpha$')
    plt.ylabel(r'Direction $\beta$')
    # plt.title('Loss Surface Contours')
    # plt.grid(True)
    plt.tight_layout()

    # Save the plot
    if save_path is None:
        # Call ensure_plot_dir to create the directory if it doesn't exist
        PLOT_DIR = ensure_plot_dir()
        save_path = os.path.join(PLOT_DIR, 'loss_surface_2d.png')
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close()

def flatten_vector_list(vector_list):
    """
    Flattens a list of tensors into a single 1D tensor.
    Args:
        vector_list (list of torch.Tensor): List of tensors to be flattened.
    Returns:
        torch.Tensor: A 1D tensor containing all the elements of the input tensors.
    """
    return torch.cat([v.view(-1) for v in vector_list])

def generate_filter_normalized_vectors(model):
    """
    Generates two orthogonal vectors in the parameter space of the model,
    normalized to the model's parameter norms.
    Args:
        model (torch.nn.Module): The PyTorch model from which to generate vectors.
    Returns:
        tuple: Two orthogonal tensors (a_flat, b_flat) representing directions in the parameter space.
    """
    params = list(model.parameters())
    
    # Generate random vectors a and b with the same shape as the model parameters
    a = [torch.randn_like(param) for param in params]
    b = [torch.randn_like(param) for param in params]
    eps = 1e-10

    for i in range(len(params)):
        model_norm = params[i].norm()
        direction_a_norm = a[i].norm()
        direction_b_norm = b[i].norm()

        # Normalize a and b to the model's parameter norms
        a[i] = a[i] * (model_norm / (eps + direction_a_norm))
        b[i] = b[i] * (model_norm / (eps + direction_b_norm))

    # Make b orthogonal to a (TO CHANGE)
    for i in range(len(params)):
        a_flat = a[i].view(-1)
        b_flat = b[i].view(-1)
        proj = (torch.dot(b_flat, a_flat) / (a_flat.norm()**2 + eps)) * a[i]
        b[i] = b[i] - proj

    a_flat = flatten_vector_list(a)
    b_flat = flatten_vector_list(b)

    return a_flat, b_flat
