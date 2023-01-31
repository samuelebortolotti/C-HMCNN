import torch.nn as nn
import torch


def get_sgd_optimizer(
    net: nn.Module, lr: float, momentum: float = 0.9
) -> torch.optim.Optimizer:
    r"""
    SGD optimizer

    Args:
        net [nn.Module]: network architecture
        momentum [float]: momentum
    Returns:
        optimizer [nn.Optimizer]
    """
    return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
