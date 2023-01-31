import torch

def get_exponential_scheduler(
    optimizer: torch.optim.Optimizer, gamma: float
) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Exponential Decay Learning Rate

    Args:
        optimizer [nn.Optimizer]
        gamma [float]: decay rate
    Returns:
        scheduler [torch.optim.lr_scheduler._LRScheduler]
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
