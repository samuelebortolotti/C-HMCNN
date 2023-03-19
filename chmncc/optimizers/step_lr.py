"""Step LR scheduler"""
import torch


def get_step_lr_scheduler(
    optimizer: torch.optim.Optimizer, step_size: int, gamma: float
) -> torch.optim.lr_scheduler._LRScheduler:
    r"""
    Get step lr scheduler

    Args:
        optimizer [nn.Optimizer]
        step_size [int]
        gamma [float]
    Returns:
        scheduler [torch.optim.lr_scheduler._LRScheduler]
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=step_size, gamma=gamma
    )
