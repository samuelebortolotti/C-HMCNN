"""Just a simple Early Stopper"""
import numpy as np


class EarlyStopper:
    """EarlyStopper class"""

    def __init__(self, patience: int = 1, min_delta: int = 0) -> None:
        """EarlyStopper initialization

        Args:
            patience [int]: patience
            min_delta [int]: min_delta for the patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """Stop the execution if the patience treshold is reached
        Args:
            validation_loss [int]: validation loss

        Returns:
            whether to stop [bool]
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
