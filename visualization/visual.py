import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import os

def ensure_plot_dir(dir_path="./exp_plots"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def flatten_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def set_flat_params(model, flat_params):
    idx = 0
    for param in model.parameters():
        numel = param.data.numel()
        param.data.copy_(flat_params[idx:idx + numel].view_as(param.data))
        idx += numel


def generate_orthogonal_directions(model):
    theta = flatten_params(model)
    dim = theta.shape[0]

    a = torch.randn(dim)
    b = torch.randn(dim)

    b = b - torch.dot(a, b) / torch.dot(a, a) * a

    a = a / torch.norm(a)
    b = b / torch.norm(b)

    return a, b


# def visual_Segment(init_model, final_model, n, loss_fn, val_loader):
#     plt.figure()
    
#     alphas = np.linspace(0, 1, n + 1)
#     losses = []

#     theta_0 = flatten_params(init_model)
#     theta_f = flatten_params(final_model)
#     currModel = copy.deepcopy(init_model)
    
#     for alpha in alphas:
#         theta = theta_0 + alpha * (theta_f - theta_0)
#         set_flat_params(currModel, theta)

#         loss, perp = loss_fn(currModel, val_loader)
#         losses.append(loss)

#     plt.plot(alphas, losses)
#     plt.xlabel('Interpolation factor (alpha)')
#     plt.ylabel('Loss')
#     plt.title('Loss along interpolation between theta_0 and theta_f')
#     plt.grid(True)
#     plt.show()
#     plt.close()   
 
def visual_segment(init_model, final_model, n, loss_fn, train_loader, val_loader, save_path=None, device='cpu'):
    """
    Plots the loss along the linear interpolation between init_model and final_model.
    If save_path is provided, saves the plot to that path (otherwise defaults to 'plots/segment_loss.png').
    """
    plt.figure()
    alphas = np.linspace(-0.5, 1.5, n + 1)
    losses_val = []
    losses_train = []
 
    theta_0 = flatten_params(init_model)
    theta_f = flatten_params(final_model)
    currModel = copy.deepcopy(init_model)
    
    for alpha in alphas:
        theta = theta_0 + alpha * (theta_f - theta_0)
        set_flat_params(currModel, theta)
        currModel.eval()

        loss_val, _ = loss_fn(currModel, val_loader, device)
        loss_train, _ = loss_fn(currModel, train_loader, device)
        losses_val.append(loss_val)
        losses_train.append(loss_train)

    # Plotting
    plt.plot(alphas, losses_train, label='Train Loss', linestyle='-', color='blue')
    plt.plot(alphas, losses_val, label='Validation Loss', linestyle='--', color='orange')
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


# def visual_2D(model, a, b, loss_fn, val_loader, n=50, range_val=1.0):
#     alphas = np.linspace(-range_val, range_val, n)
#     betas = np.linspace(-range_val, range_val, n)
#     loss_grid = np.zeros((n, n))

#     device = next(model.parameters()).device

#     # Ensure a, b are on same device as model
#     a = a.to(device)
#     b = b.to(device)

#     theta_0 = flatten_params(model).to(device)
#     currModel = copy.deepcopy(model).to(device)

#     for i, alpha in enumerate(alphas):
#         for j, beta in enumerate(betas):
#             theta = theta_0 + alpha * a + beta * b
#             set_flat_params(currModel, theta)
#             currModel.to(device)  # Just to be safe

#             loss, _ = loss_fn(currModel, val_loader)
#             loss_grid[i, j] = loss

#     A, B = np.meshgrid(alphas, betas)
#     plt.figure(figsize=(8, 6))
    
#     # Filled contour plot
#     cp = plt.contourf(A, B, loss_grid.T, levels=50)
    
#     # Add contour lines
#     lines = plt.contour(A, B, loss_grid.T, levels=20, colors='black', linewidths=0.5)
#     plt.clabel(lines, inline=True, fontsize=8)
    
#     plt.colorbar(cp)
#     plt.xlabel('Direction a (alpha)')
#     plt.ylabel('Direction b (beta)')
#     plt.title('2D Loss Surface with Contour Lines')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     plt.close()

def visual_2D(model, a, b, loss_fn, train_loader, n=50, range_val=1.0, save_path=None):
    """
    Computes and plots a 2D loss surface in directions a and b.
    If save_path is provided, saves the contour plot (otherwise defaults to 'plots/loss_surface_2d.png').
    """
    alphas = np.linspace(-range_val, range_val, n)
    betas = np.linspace(-range_val, range_val, n)
    loss_grid = np.zeros((n, n))

    device = next(model.parameters()).device

    # Ensure a, b are on same device as model
    a = a.to(device)
    b = b.to(device)

    theta_0 = flatten_params(model).to(device)
    currModel = copy.deepcopy(model).to(device)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            theta = theta_0 + alpha * a + beta * b
            set_flat_params(currModel, theta)
            currModel.eval()

            with torch.no_grad():
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
    return torch.cat([v.view(-1) for v in vector_list])

def generate_filter_normalized_vectors(model):
    params = list(model.parameters())
    a = [torch.randn_like(param) for param in params]
    b = [torch.randn_like(param) for param in params]
    eps = 1e-10

    for i in range(len(params)):
        model_norm = params[i].norm()
        direction_a_norm = a[i].norm()
        direction_b_norm = b[i].norm()
        a[i] = a[i] * (model_norm / (eps + direction_a_norm))
        b[i] = b[i] * (model_norm / (eps + direction_b_norm))

    # make b orthogonal to a (TO CHANGE)
    for i in range(len(params)):
        a_flat = a[i].view(-1)
        b_flat = b[i].view(-1)
        proj = (torch.dot(b_flat, a_flat) / (a_flat.norm()**2 + eps)) * a[i]
        b[i] = b[i] - proj

    a_flat = flatten_vector_list(a)
    b_flat = flatten_vector_list(b)

    return a_flat, b_flat
