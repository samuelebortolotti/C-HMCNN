"""Adam module"""
import torch.nn as nn
import torch


def get_adam_optimizer(
    net: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    r"""
    ADAM optimizer

    Args:
        net [nn.Module]: network architecture
        lr [float]: learning rate
        weight_decay [float]: weight_decay
    Returns:
        optimizer [nn.Optimizer]
    """
    return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
