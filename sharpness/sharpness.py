import torch 
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

@torch.no_grad() # avoid gradient tracking
def evaluate_loss(model, dataloader, loss_fn, device):
    """Compute the average loss of the model.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset. 
        loss_fn: Loss function to use. 
        device: Device to perform computations on. ('cuda' or 'cpu')
    Returns:
        average_loss: The average loss of the model.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = [x.to(device) for x in batch]
        inputs = batch[0]
        attention_mask = batch[1]

        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1))
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

    return total_loss / total_tokens

## Epsilon sharpness, based on https://arxiv.org/abs/2004.01461
def compute_epsilon_sharpness(
    model, 
    dataloader, 
    loss_fn, 
    epsilon=1e-3, 
    num_samples=10, 
    device='cuda'
):
    """
    Compute the sharpness of the model by evaluating the loss on perturbed parameters.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        loss_fn: Loss function to use.
        epsilon: Perturbation size.
        num_samples: Number of samples to average over.
        device: Device to perform computations on. ('cuda' or 'cpu')
    Returns:
        sharpness: The maximum relative increase in loss due to perturbations.
        base_loss: The base loss of the model.
    """
    model.eval()
    theta = parameters_to_vector(model.parameters()).detach().to(device)
    base_loss = evaluate_loss(model, dataloader, loss_fn, device)

    sharpness_values = []
    for _ in range(num_samples):
        delta = torch.randn_like(theta)
        delta = epsilon * delta / delta.norm()

        perturbed_theta = theta + delta
        vector_to_parameters(perturbed_theta, model.parameters())

        perturbed_loss = evaluate_loss(model, dataloader, loss_fn, device)
        rel_increase = ((perturbed_loss - base_loss) / (1 + base_loss)) * 100
        sharpness_values.append(rel_increase)

    # Restore original weights
    vector_to_parameters(theta, model.parameters())
    return max(sharpness_values), base_loss

## Hessian Eigenvalues sharpness
def hessian_vector_product(loss, params, v):
    """Compute the Hessian-vector product.
    Args:
        loss: The loss value.
        params: The model parameters.
        v: The vector to compute the Hessian-vector product with.
    Returns:
        hessian_vector: The Hessian-vector product.
    """
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vector = parameters_to_vector(grads)
    grad_dot_v = torch.dot(grad_vector, v)
    hessian_vector = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return parameters_to_vector(hessian_vector)

def batch_averaged_hvp(model, dataloader, params, v, device, num_batches=3):
    """Compute the batch-averaged Hessian-vector product.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        params: The model parameters.
        v: The vector to compute the Hessian-vector product with.
        device: Device to perform computations on. ('cuda' or 'cpu')
        num_batches: Number of batches to average over.
    Returns:
        hv_acc: The averaged Hessian-vector product.
    """
    hv_acc = torch.zeros_like(v, device=device) #Batches Hv
    it = iter(dataloader)
    for _ in range(num_batches):
        try:
            inputs, attention_mask = next(it)
        except StopIteration:
            it = iter(dataloader) # reset iterator
            inputs, attention_mask = next(it)
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        
        # Had issues with gradient checkpointing and flash attention so disabled it
        with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False
            ):
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=inputs
            )
            loss = outputs.loss
        hv_acc += hessian_vector_product(loss, params, v)
    return hv_acc / num_batches

def power_iteration_hessian(model, dataloader, device,
                            num_iters=30, num_batches=3, tol=1e-2):
    """Compute the largest eigenvalue of the Hessian using power iteration.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        device: Device to perform computations on. ('cuda' or 'cpu')
        num_iters: Max number of iterations for power iteration.
        num_batches: Number of batches to average over per iteration.
        tol: Tolerance for convergence of eigenvalue.
    Returns:
        lambda_max: The largest eigenvalue of the Hessian.
        v: The corresponding eigenvector.
    """
    model.eval()
    params = list(model.parameters())

    # Normalize the parameters layer by layer
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() > 1:
            norm = param.data.norm()
            if norm > 0:
                param.data = param.data / norm

    dim = sum(p.numel() for p in params)
    v = torch.randn(dim, device=device)
    v /= v.norm()

    prev_hv_norm = None
    for i in range(num_iters):
        hv = batch_averaged_hvp(model, dataloader, params, v, device, num_batches)
        hv_norm = hv.norm()
        v = hv / (hv_norm + 1e-12)

        if prev_hv_norm is not None and abs(hv_norm.item() - prev_hv_norm) < tol:
            print(f"Converged at iteration {i + 1} with eigenvalue ≈ {hv_norm.item():.4f}")
            break

        prev_hv_norm = hv_norm.item()

    return hv_norm.item(), v


def compute_epsilon_hessian_sharpness(
    model, 
    dataloader, 
    loss_fn, 
    v,
    epsilon=1e-3, 
    num_samples=10, 
    device='cuda'
):
    """
    Compute the sharpness of the model by evaluating the loss on perturbed parameters.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        loss_fn: Loss function to use.
        v: The largest eigenvector of the Hessian.
        epsilon: Perturbation size.
        num_samples: Number of samples to average over.
        device: Device to perform computations on. ('cuda' or 'cpu')
    Returns:
        sharpness: The maximum relative increase in loss due to perturbations.
        base_loss: The base loss of the model.
    """
    model.eval()
    theta = parameters_to_vector(model.parameters()).detach().to(device)
    base_loss = evaluate_loss(model, dataloader, loss_fn, device)

    sharpness_values = []
    for _ in range(num_samples):
        delta = epsilon * v 

        perturbed_theta = theta + delta
        vector_to_parameters(perturbed_theta, model.parameters())

        perturbed_loss = evaluate_loss(model, dataloader, loss_fn, device)
        rel_increase = ((perturbed_loss - base_loss) / (1 + base_loss)) * 100
        sharpness_values.append(rel_increase)

    # Restore original weights
    vector_to_parameters(theta, model.parameters())
    return max(sharpness_values), base_loss
