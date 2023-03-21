"""MLP module"""
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
        self,
        R: torch.Tensor,
        num_out_logits: int = 20,
        constrained_layer: bool = True,
        superclasses_number: int = 20,
        use_softmax: bool = False,
        img_width: int = 32,
        img_height: int = 32,
        channels: int = 3,
        dropout: float = 0.10,
    ) -> None:
        r"""
        Initialize the MLP model
        Default:
            num_out_logits [int] = 20
        Args:
            R [torch.Tensor]: adjacency matrix
            num_out_logits [int]: number of output logits
            constrained_layer: [bool]: whether to use the constrained output approach from Giunchiglia et al.
            superclasses_number [int]: superclass number
            use_softmax: [bool]: use softmax
            img_width [int] = 32: width of the images
            img_height [int] = 32: height of the images
            channels [int] = 3: channel numbers
            dropout [float] = 0.10: dropout rate
        """
        super().__init__()
        self.R = R  # matrix of the hierarchy
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.superclasses_number = superclasses_number
        self.use_softmax = use_softmax
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.img_height * self.img_width * self.channels,
                out_features=1536,
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=1536, out_features=768),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=384),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=num_out_logits),
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

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

        if self.use_softmax:
            # root prediction
            x[:, 0] = self.sigmoid(x[:, 0])
            # parent prediction
            x[:, 1 : self.superclasses_number + 1] = self.softmax(
                x[:, 1 : self.superclasses_number + 1]
            )
            # children prediction
            x[:, self.superclasses_number + 1 :] = self.softmax(
                x[:, self.superclasses_number + 1 :]
            )
        else:
            x = self.sigmoid(x)

        # if we are in trainining, no need to enforce the herarchy constraint
        # or if the constrained layer is unwanted
        if not self.constrained_layer or self.training:
            constrained_out = x
        else:
            #  print("X before:", x)
            constrained_out = get_constr_out(
                x, self.R
            )  # in validation and test: herarchy set
            #  print("X after", constrained_out)
        return constrained_out
