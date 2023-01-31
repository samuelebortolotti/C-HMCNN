import torch

def get_exponential_scheduler(
    optimizer: torch.optim.Adam, gamma: float
) -> torch.optim.Optimizer:
    r"""
    Exponential Decay Learning Rate

    Args:
        optimizer [nn.Optimizer]
        gamma [float]: decay rate
    Returns:
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
