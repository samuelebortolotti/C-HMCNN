import torch.nn as nn
import torch
import torch.nn.functional as F
from chmncc.utils import get_constr_out


class DummyCNN(nn.Module):
    """
    Initialize the DummyCNN model.
    It is just the CNN of the basic PyTorch tutorial: 2 conv and 3 fully connected
    Args:
        R [torch.Tensor]: adjacency matrix
        num_out_logits [int]: number of output logits
        constrained_layer: [bool]: whether to use the constrained output approach from Giunchiglia et al.
    """

    def __init__(
        self, R: torch.Tensor, num_out_logits: int = 20, constrained_layer: bool = True
    ) -> None:
        super().__init__()
        self.R = R  # matrix of the hierarchy
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_out_logits)
        self.activation = nn.Sigmoid()

        # set whether to use the constrained layer or not
        self.constrained_layer = constrained_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x [torch.Tensor]: source sample
        Returns:
            constrained_out [torch.Tensor]: constrained_out
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)

        # if we are in trainining, no need to enforce the herarchy constraint
        # or if the constrained layer is unwanted
        if not self.constrained_layer or self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(
                x, self.R
            )  # in validation and test: herarchy set
        return constrained_out
