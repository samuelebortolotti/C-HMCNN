"""Lenet module"""
import torch.nn as nn
import torch
from chmncc.utils import get_constr_out


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
        R: torch.Tensor,
        num_out_logits: int = 20,
        constrained_layer: bool = True,
    ) -> None:
        r"""
        Initialize the LeNet5 model
        Default:
            num_out_logits [int] = 20
        Args:
            R [torch.Tensor]: adjacency matrix
            num_out_logits [int]: number of output logits
            constrained_layer: [bool]: whether to use the constrained output approach from Giunchiglia et al.
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
            nn.Sigmoid(),
        )

        # set whether to use the constrained layer or not
        self.constrained_layer = constrained_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method
        Args:
            x [torch.Tensor]: source sample
        Returns:
            constrained_out [torch.Tensor]: constrained_out
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        # if we are in trainining, no need to enforce the herarchy constraint
        # or if the constrained layer is unwanted
        if not self.constrained_layer or self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(
                x, self.R
            )  # in validation and test: herarchy set
        return constrained_out
