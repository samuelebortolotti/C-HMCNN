"""Module which deals with the explaination we can retrieve out of the images"""
import torch
import tqdm


def compute_integrated_gradient(
    batch_x: torch.Tensor, batch_blank: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    """Integrated gradients computation
    Implementation taken from https://github.com/CVxTz/IntegratedGradientsPytorch/blob/main/code/mlp_gradient.py

    Args:
        batch_x [torch.Tensor] data instances batch
        batch_blank [torch.Tensor] batch of zeros
        model [torch.nn.Module] network
    Returns:
        integrated_gradients [torch.Tensor] integrated gradients
    """
    mean_grad = 0
    n = 10

    for i in tqdm.tqdm(range(1, n + 1), desc="Computing the integrated gradient"):
        x = batch_blank + i / n * (batch_x - batch_blank)
        #  x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
        )
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients[0]


def output_gradients(inputs: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Compute the gradients with respect to the input

    Args:
        inputs [torch.Tensor] input tensor
        preds [torch.Tensor] output tensor
    Returns:
        grad [torch.Tensor] gradients of the output with respect to the input
    """
    return torch.autograd.grad(
        outputs=preds,
        inputs=inputs,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True,
    )[0]
