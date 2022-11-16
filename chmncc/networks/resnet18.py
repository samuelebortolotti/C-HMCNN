import torch
import torch.nn as nn
from torchvision.models import resnet18
from chmncc.utils import get_constr_out

class ResNet18(nn.Module):
    r"""
    Original ResNet18 architecture, which is taken directly from the torchvision
    models
    """

    def __init__(
        self,
        R: torch.tensor,
        num_classes: int = 20,
        pretrained: bool = False
    ) -> None:
        r"""
        Initialize the basic ResNet18 architecture
        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        Args:
        - num_classes [int]: number of classes [used in the last layer]
        - pretrained [bool]: whether to pretrain the model or not
        """
        super(ResNet18, self).__init__()
        self.R = R  # matrix of the hierarchy
        # Take the resNet18 module and discard the last layer
        if pretrained:
            backbone = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        else:
            backbone = resnet18()
        features = nn.ModuleList(backbone.children())[:-1]

        # set the ResNet18 backbone as feature extractor
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        r"""
        Forward method
        Args:
        - x [torch.Tensor]: source sample
        - device [str]: output device
        Returns:
        - prediction [torch.Tensor]: prediction
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.training:
            constrained_out = x
        else:
            x_copy = x.to('cpu')
            constrained_out = get_constr_out(
                x_copy, self.R
            )  # in validation and test: herarchy set
            constrained_out = constrained_out.to(device)
        return constrained_out
