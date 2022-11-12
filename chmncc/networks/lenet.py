import torch.nn as nn
import torch
from chmncc.utils import get_constr_out
from torch.nn.modules.activation import Sigmoid


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
            # input channel = 1, output channels = 6, kernel size = 5
            # input image size = (28, 28), image output size = (24, 24)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            # input channel = 6, output channels = 16, kernel size = 5
            # input image size = (12, 12), output image size = (8, 8)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
        )
        self.classifier = nn.Sequential(
            # input dim = 4 * 4 * 16 ( H x W x C), output dim = 120
            nn.Linear(in_features=4 * 4 * 16, out_features=120),
            # input dim = 120, output dim = 84
            nn.Linear(in_features=120, out_features=84),
            # input dim = 84, output dim = 10
            nn.Linear(in_features=84, out_features=num_out_logits),
            # sigmoid in the last layer
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method
        Args:
            - x [torch.Tensor]: source sample
        Returns:
            - constrained_out [torch.Tensor]: constrained_out
        """
        x = self.features(x)
        x = self.classifier(x)

        # if we are in trainining, no need to enforce the herarchy constraint
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(
                x, self.R
            )  # in validation and test: herarchy set
        return constrained_out
