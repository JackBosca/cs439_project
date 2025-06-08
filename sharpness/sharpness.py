import torch 
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from optimization.evaluation import compute_loss

# @torch.no_grad() # avoid gradient tracking
# def evaluate_loss(model, dataloader, loss_fn, device):
#     """Compute the average loss of the model.
#     Args:
#         model: The model to evaluate. (e.g., DistilGPT2)
#         dataloader: DataLoader for the dataset. 
#         loss_fn: Loss function to use. 
#         device: Device to perform computations on. ('cuda' or 'cpu')
#     Returns:
#         average_loss: The average loss of the model.
#     """
#     model.eval()
#     total_loss = 0
#     total_tokens = 0
#     for batch in tqdm(dataloader, desc="Evaluating", leave=False):
#         batch = [x.to(device) for x in batch]
#         inputs = batch[0]
#         attention_mask = batch[1]

#         inputs = inputs.to(device)
#         attention_mask = attention_mask.to(device)
#         if attention_mask is not None:
#             attention_mask = attention_mask.to(device)

#         outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
#         loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.view(-1))
#         total_loss += loss.item() * inputs.numel()
#         total_tokens += inputs.numel()

#     return total_loss / total_tokens

# ## Epsilon sharpness, based on https://arxiv.org/abs/2004.01461
# def compute_epsilon_sharpness(
#     model, 
#     dataloader, 
#     loss_fn, 
#     epsilon=1e-3, 
#     num_samples=10, 
#     device='cuda'
# ):
#     """
#     Compute the sharpness of the model by evaluating the loss on perturbed parameters.
#     Args:
#         model: The model to evaluate. (e.g., DistilGPT2)
#         dataloader: DataLoader for the dataset.
#         loss_fn: Loss function to use.
#         epsilon: Perturbation size.
#         num_samples: Number of samples to average over.
#         device: Device to perform computations on. ('cuda' or 'cpu')
#     Returns:
#         sharpness: The maximum relative increase in loss due to perturbations.
#         base_loss: The base loss of the model.
#     """
#     model.eval()
#     theta = parameters_to_vector(model.parameters()).detach().to(device)
#     base_loss = evaluate_loss(model, dataloader, loss_fn, device)

#     sharpness_values = []
#     for _ in range(num_samples):
#         delta = torch.randn_like(theta)
#         delta = epsilon * delta / delta.norm()

#         perturbed_theta = theta + delta
#         vector_to_parameters(perturbed_theta, model.parameters())

#         perturbed_loss = evaluate_loss(model, dataloader, loss_fn, device)
#         rel_increase = ((perturbed_loss - base_loss) / (1 + base_loss)) * 100
#         sharpness_values.append(rel_increase)

#     # Restore original weights
#     vector_to_parameters(theta, model.parameters())
#     return max(sharpness_values), base_loss

def hessian_vector_product(loss, params, v):
    """Compute the Hessian-vector product.
    Args:
        loss: The loss value.
        params: The model parameters.
        v: The vector to compute the Hessian-vector product with.
    Returns:
        hessian_vector: The Hessian-vector product.
    """
    # Compute the gradient of the loss with respect to the parameters
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Convert the gradients to a vector
    grad_vector = parameters_to_vector(grads)

    # Compute the dot product of the gradient vector and the vector v
    grad_dot_v = torch.dot(grad_vector, v)

    # Compute the Hessian-vector product
    # hessian_vector = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    hessian_vector = torch.autograd.grad(grad_dot_v, params)

    # Convert the Hessian-vector product to a vector
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
    # Initialize the Hessian-vector product accumulator
    hv_acc = torch.zeros_like(v, device=device)
    it = iter(dataloader)

    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            # If we reach the end of the iterator, reset it
            it = iter(dataloader)
            batch = next(it)

        # Handle both tuple and non-tuple batches and move to device
        if isinstance(batch, (tuple, list)):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None
        else:
            inputs = batch.to(device)
            attention_mask = None
        
        # Disable flash attention and math for the model
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

            # Compute the loss
            loss = outputs.loss

        # Compute the Hessian-vector product
        hv_acc += hessian_vector_product(loss, params, v)

    # Average the Hessian-vector product over the number of batches
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
    # Move model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    params = list(model.parameters())

    # Normalize the parameters layer by layer
    for _, param in model.named_parameters():
        if param.requires_grad and param.dim() > 1:
            norm = param.data.norm()
            if norm > 0:
                param.data = param.data / norm

    # Sum the number of elements in all parameters
    dim = sum(p.numel() for p in params)

    # Initialize a random vector
    v = torch.randn(dim, device=device)
    v /= v.norm()

    prev_hv_norm = None
    for i in tqdm(range(num_iters), desc="Power Iteration"):
        # Compute the Hessian-vector product and normalize the vector
        hv = batch_averaged_hvp(model, dataloader, params, v, device, num_batches)

        with torch.no_grad():
            hv_norm = hv.norm()
            v = hv / (hv_norm + 1e-12)

        # Check for convergence
        if prev_hv_norm is not None and abs(hv_norm.item() - prev_hv_norm) < tol:
            print(f"Converged at iteration {i + 1} with eigenvalue ≈ {hv_norm.item():.4f} \n")
            break

        # Save the current Hessian-vector product norm for the next iteration
        prev_hv_norm = hv_norm.item()

    return hv_norm.item(), v

def compute_epsilon_hessian_sharpness(model, dataloader, loss_fn, v,
                                      epsilon=1e-3, num_samples=1, rand_dir=False, base_loss=None, device='cpu'):
    """
    Compute the sharpness of the model by evaluating the loss on perturbed parameters.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        loss_fn: Loss function to use.
        v: The largest eigenvector of the Hessian.
        epsilon: Perturbation size.
        num_samples: Number of samples to average over.
        rand_dir: If True, use a random direction instead of the largest eigenvector.
        base_loss: The base loss of the model.
        device: Device to perform computations on. ('cuda' or 'cpu')
    Returns:
        sharpness: The maximum relative increase in loss due to perturbations.
        base_loss: The base loss of the model.
    """
    model.eval()

    # Convert the model parameters to a vector
    theta = parameters_to_vector(model.parameters()).detach().to(device)

    # Compute the base loss
    if base_loss is None:
        base_loss = compute_loss(model, dataloader, device, loss_fn=loss_fn, return_perplexity=False, show_progress=False)

    sharpness_values = []
    for _ in tqdm(range(num_samples), desc="Perturbation samples"):
        # Generate a random perturbation vector
        if rand_dir:
            v = torch.randn_like(theta)
            v /= v.norm()
        else:
            # Use the largest eigenvector of the Hessian
            v = v / v.norm()
        
        # Perturb the model parameters
        delta = epsilon * v 
        perturbed_theta = theta + delta

        # Convert the perturbed vector back to model parameters
        vector_to_parameters(perturbed_theta, model.parameters())

        # Compute the perturbed loss
        with torch.no_grad():
            perturbed_loss = compute_loss(model, dataloader, device, loss_fn=loss_fn, 
                                          return_perplexity=False, max_batches=3, show_progress=False)

        # Compute the relative increase in loss and append to the list
        rel_increase = ((perturbed_loss - base_loss) /  base_loss) * 100
        sharpness_values.append(rel_increase)

    # Restore original weights
    vector_to_parameters(theta, model.parameters())

    # Return the maximum sharpness value and the base loss
    return max(sharpness_values), base_loss

def check_sharpness_approximation(model, dataloader, v, lambda_max, epsilon=1e-3, device='cpu'):
    """
    Check the sharpness approximation using the largest eigenvector of the Hessian.
    Args:
        model: The model to evaluate. (e.g., DistilGPT2)
        dataloader: DataLoader for the dataset.
        v: The largest eigenvector of the Hessian.
        lambda_max: The largest eigenvalue of the Hessian.
        epsilon: Perturbation size.
        device: Device to perform computations on. ('cuda' or 'cpu')
    Returns:
        measured_rel_increase: The measured relative increase in loss.
        predicted_rel_increase: The predicted relative increase in loss.
        linear_term: The linear term in the approximation.
        quad_term: The quadratic term in the approximation.
    """
    model.eval()
    
    # Flatten current parameters
    theta = parameters_to_vector(model.parameters()).detach().to(device)

    # Compute base loss and gradient
    inputs_list = []
    attention_masks_list = []

    # Just 1 batch is enough for a check
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None
        else:
            inputs = batch.to(device)
            attention_mask = None
        inputs_list.append(inputs)
        attention_masks_list.append(attention_mask)
        break  # only one batch

    inputs = inputs_list[0]
    attention_mask = attention_masks_list[0]

    # Forward and compute loss
    outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
    loss = outputs.loss
    base_loss = loss.item()

    # Compute gradient
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad_vec = parameters_to_vector(grads).detach()

    # Perturb along v
    delta = epsilon * v
    perturbed_theta = theta + delta
    vector_to_parameters(perturbed_theta, model.parameters())

    # Compute perturbed loss
    with torch.no_grad():
        outputs_perturbed = model(inputs, attention_mask=attention_mask, labels=inputs)
        perturbed_loss = outputs_perturbed.loss.item()

    # Restore original parameters
    vector_to_parameters(theta, model.parameters())

    # Compute terms
    linear_term = grad_vec @ delta
    quad_term = 0.5 * epsilon**2 * lambda_max

    measured_rel_increase = (perturbed_loss - base_loss) / base_loss
    predicted_rel_increase = (linear_term + quad_term) / base_loss

    print(f"Measured relative loss increase: {measured_rel_increase * 100:.2f}%")
    print(f"Predicted relative loss increase: {predicted_rel_increase.item() * 100:.2f}%")
    print(f"Linear term: {linear_term.item():.4e}")
    print(f"Quadratic term (0.5 * eps^2 * lambda_max): {quad_term:.4e}")

    return {
        "measured": measured_rel_increase,
        "predicted": predicted_rel_increase.item(),
        "linear_term": linear_term.item(),
        "quad_term": quad_term
    }
