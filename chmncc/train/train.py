import torch
import torch.nn as nn
import torch.nn.functional as F
from chmncc.utils import dotdict
import numpy as np
from typing import Tuple
import tqdm
from chmncc.utils import get_constr_out


def training_step(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train: dotdict,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cost_function: torch.nn.modules.loss.BCELoss,
    title: str,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Training step of the network. It works both for our approach and for the one of
    Giunchiglia et al.

    Args:
        net [nn.Module] network on device
        train_loader [torch.utils.data.DataLoader] training data loader
        train [dotdict] training set dictionary
        R [torch.Tensor] adjency matrix
        optimizer [torch.Tensor] adjency matrix
        cost_function [torch.nn.modules.loss.BCELoss] Binary Cross Entropy loss
        title [str] title of the experiment
        device [str]: on which device to run the experiment [default: cuda]

    Returns:
        cumulative loss [float]
        accuracy [float] in percentange
    """
    total_train = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network to training mode
    net.train()

    # iterate over the training set
    for _, inputs in tqdm.tqdm(enumerate(train_loader), desc=title):

        # according to the Giunchiglia dataset
        inputs, label = inputs

        # load data into device
        inputs = inputs.to(device)
        label = label.to(device)

        # gradients reset
        optimizer.zero_grad()

        # output
        outputs = net(inputs.float())

        # general prediction loss computation
        # MCLoss (their loss)
        constr_output = get_constr_out(outputs, R)
        train_output = label * outputs.double()
        train_output = get_constr_out(train_output, R)
        train_output = (1 - label) * constr_output.double() + label * train_output
        # get the loss masking the prediction on the root -> confunder
        loss = cost_function(train_output[:, train.to_eval], label[:, train.to_eval])
        cumulative_loss += loss.item()

        predicted = constr_output.data > 0.5

        # fetch prediction and loss value
        total_train += label.shape[0] * label.shape[1]

        # compute training accuracy
        cumulative_accuracy += (predicted == label.byte()).sum().item()

        # backward pass
        loss.backward()
        # optimizer
        optimizer.step()

    return cumulative_loss / len(train_loader), cumulative_accuracy / total_train * 100
