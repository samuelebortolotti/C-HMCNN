import torch.nn as nn
import torch
from chmncc.utils import get_constr_out


class Flatten(nn.Module):
    """Flatten layer"""

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LeNet7(nn.Module):
    r"""
    LeNet5 architecture with Giunchiglia et al layer
    """

    def __init__(
        self,
        R: torch.Tensor,
        num_out_logits: int = 20,
        constrained_layer: bool = True,
        force_superclass_prediction: bool = False,
        superclasses_number: int = 20,
    ) -> None:
        r"""
        Initialize the LeNet5 model
        Default:
            num_out_logits [int] = 20
        Args:
            R [torch.Tensor]: adjacency matrix
            num_out_logits [int]: number of output logits
            constrained_layer [bool]: whether to use the constrained output approach from Giunchiglia et al.
            force_superclass_prediction [bool]: force superclass prediction
            superclasses_number [int]: superclass number
        """
        super().__init__()
        self.R = R  # matrix of the hierarchy
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_out_logits),
        )
        self.sigmoid = nn.Sigmoid()

        self.force_superclass_prediction = force_superclass_prediction
        self.superclasses_number = superclasses_number

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
        if self.force_superclass_prediction:
            import torch.nn.functional as F

            x[:, 1 : self.superclasses_number + 1] = F.softmax(
                x[:, 1 : self.superclasses_number + 1], dim=1
            )
            x[:, self.superclasses_number + 1 :] = torch.sigmoid(
                x[:, self.superclasses_number + 1 :]
            )
            x[:, 0] = torch.sigmoid(x[:, 0])
        else:
            x = self.sigmoid(x)

        # if we are in trainining, no need to enforce the herarchy constraint
        # or if the constrained layer is unwanted
        if not self.constrained_layer or self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(
                x, self.R
            )  # in validation and test: herarchy set
        return constrained_out