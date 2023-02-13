import torch.nn as nn
import torch
from chmncc.utils import get_constr_out

class Flatten(nn.Module):
    """Flatten layer"""

    def forward(self, x):
        return x.view(x.shape[0], -1)

class MLP(nn.Module):
    r"""
    MLP architecture with Giunchiglia et al layer
    """

    def __init__(
        self, R: torch.Tensor, num_out_logits: int = 20, constrained_layer: bool = True, img_width: int = 32, img_height: int = 32, channels: int = 3
    ) -> None:
        r"""
        Initialize the MLP model
        Default:
            num_out_logits [int] = 20
        Args:
            R [torch.Tensor]: adjacency matrix
            num_out_logits [int]: number of output logits
            constrained_layer: [bool]: whether to use the constrained output approach from Giunchiglia et al.
        """
        super().__init__()
        self.R = R  # matrix of the hierarchy
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.img_height * self.img_width * self.channels, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_out_logits),
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
        x = x.view(-1, self.img_height * self.img_width * self.channels)
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
