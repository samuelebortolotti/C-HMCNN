import torch
import tqdm

# Implementation taken from https://github.com/CVxTz/IntegratedGradientsPytorch/blob/main/code/mlp_gradient.py
def compute_integrated_gradient(batch_x, batch_blank, model):
    mean_grad = 0
    n = 100

    for i in tqdm.tqdm(range(1, n + 1)):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)
        (grad,) = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
        )
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients, mean_grad


def output_gradients(inputs, preds):
    r"""Compute the gradients with respect to the input"""
    return torch.autograd.grad(
        outputs=preds,
        inputs=inputs,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True,
    )[0]
