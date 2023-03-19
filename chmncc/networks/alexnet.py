"""AlexNet module"""
import torch.nn as nn
import torch
from chmncc.utils import get_constr_out


class AlexNet(nn.Module):
    """
    AlexNet architecture taken from the PyTorch source code.
    The reference is taken from
    [link]: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
    """

    def __init__(
        self,
        R: torch.Tensor,
        num_out_logits: int = 20,
        constrained_layer: bool = True,
        dropout: float = 0,
        pretrained: bool = False,
    ) -> None:
        r"""
        Initialize the AlexNet model
        Args:
            R [torch.Tensor]: adjacency matrix
            num_out_logits [int] = 20
            constrained_layer: [bool]: whether to use the constrained output approach from Giunchiglia et al.
            dropout [float] = 0
            pretrained [bool] = False
        """
        super().__init__()
        self.R = R  # matrix of the hierarchy

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # number of classes as output channel of the last fc layer
            nn.Linear(4096, num_out_logits),
        )
        self.activation = nn.Sigmoid()

        # set whether to use the constrained layer or not
        self.constrained_layer = constrained_layer

        # automatic pretrained model
        if pretrained:
            # url of the AlexNet weights
            from torchvision.models import alexnet as anet

            if pretrained:
                backbone = anet(weights="AlexNet_Weights.IMAGENET1K_V1")
            else:
                backbone = anet()

            # load the weights
            state_dict = backbone.state_dict()
            # remove the last layer weights
            state_dict["classifier.6.weight"] = self.state_dict()["classifier.6.weight"]
            state_dict["classifier.6.bias"] = self.state_dict()["classifier.6.bias"]
            # load the weights
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method
        Args:
            x [torch.Tensor]: source sample
        Returns:
            prediction [torch.Tensor]: prediction
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
