import torch.nn as nn
import torch
from chmncc.utils import get_constr_out
from torch.nn.modules.activation import Sigmoid


class Flatten(nn.Module):
    """Flatten layer"""

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LeNet5(nn.Module):
    r"""
    LeNet5 architecture with Giunchiglia et al layer
    """

    def __init__(
        self,
        R: torch.tensor,
        num_out_logits: int = 20,
    ) -> None:
        r"""
        Initialize the LeNet5 model
        Default:
        - num_out_logits [int] = 20
        Args:
        - num_out_logits [int]: number of output logits
        """
        super().__init__()
        self.R = R  # matrix of the hierarchy
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=5 * 5 * 16, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_out_logits),
            #  nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        r"""
        Forward method
        Args:
            - x [torch.Tensor]: source sample
        Returns:
            - constrained_out [torch.Tensor]: constrained_out
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        # if we are in trainining, no need to enforce the herarchy constraint
        if self.training:
            constrained_out = x
        else:
            x_copy = x.to('cpu')
            constrained_out = get_constr_out(
                x_copy, self.R
            )  # in validation and test: herarchy set
            constrained_out = constrained_out.to(device)
        return constrained_out
