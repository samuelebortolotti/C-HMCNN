"""Optimizers module
It deals with all the optimizers we have employed or with the approaches we have experimented
"""
from .adam import get_adam_optimizer
from .exponential import get_exponential_scheduler
from .reduce_lr_on_plateau import get_plateau_scheduler
from .sgd import get_sgd_optimizer
