import torch.nn as nn
import torch
from chmncc.utils import get_constr_out
from chmncc.config.old_config import hidden_dims, input_dims, output_dims
from typing import Dict


def initializeConstrainedFFNNModel(
    dataset_name: str,
    data: str,
    ontology: str,
    R: torch.tensor,
    hyperparams: Dict[str, float],
):
    """Initialize the ConstrainedFFNNModel as described in the
    Giunchiglia et al paper

    Args:
        dataset_name [str]: name of the dataset (data + ontology)
        data [str]: name of the data
        ontology [str]: name of the ontology
        R [torch.tensor]: adjacency matrix
    """
    if "GO" in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # return the model
    return ConstrainedFFNNModel(
        input_dims[data],
        hidden_dims[ontology][data],
        output_dims[ontology][data] + num_to_skip,
        hyperparams,
        R,
    )


class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h) model:
    during training it returns the not-constrained output that is then passed to MCLoss"""

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        """Constructor of the ConstrainedFFNNModel

        Args:
            input_dim [int] dimension of the input layer
            hidden_dim [int] dimension of the hidden layer
            output_dim [int] dimension of the output layer
            hyperparams [Dict[float]] dictionary of the hyperparameters
            R [torch.tensor] matrix collecting the hierarchical relationship
        """
        super(ConstrainedFFNNModel, self).__init__()

        self.nb_layers = hyperparams["num_layers"]
        self.R = R  # matrix of the hierarchy

        fc = []
        # append linear layers according to the number of hidden layers needed
        for i in range(self.nb_layers):
            if i == 0:  # input
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:  # output
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:  # hidden
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)  # module list

        # dropout rate
        self.drop = nn.Dropout(hyperparams["dropout"])

        # sigmoid activation
        self.sigmoid = nn.Sigmoid()
        if hyperparams["non_lin"] == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        """Forward method of the model

        Args:
            x [torch.tensor]: input tensor

        Returns:
            constrained_out [torch.tensor]: constrained ouput
        """
        # loop over all the layers, on the last one employ the sigmoid activation
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                # apply the layer and the dropout
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        # if we are in trainining, no need to enforce the herarchy constraint
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(
                x, self.R
            )  # in validation and test: herarchy set
        return constrained_out
