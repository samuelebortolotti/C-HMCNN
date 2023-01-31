import torch

def get_plateau_scheduler(
    optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Get Reduce on Plateau scheduler

    Args:
        optimizer [nn.Optimizer]
    Returns:
        scheduler [torch.optim.lr_scheduler._LRScheduler]
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

