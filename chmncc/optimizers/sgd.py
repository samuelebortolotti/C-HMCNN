"""Stochastic gradient descend scheduler"""
import torch.nn as nn
import torch
from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction


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


def get_sdg_optimizer_with_gate(
    net: nn.Module, gate: DenseGatingFunction, lr: float, weight_decay: float
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
    return torch.optim.SGD(
        list(net.parameters()) + list(gate.parameters()), lr=lr, momentum=0.9
    )
