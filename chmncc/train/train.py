import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import tqdm
import torch.nn.functional as F
from chmncc.utils import get_constr_out


def training_step(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train,
    R: torch.tensor,
    writer,
    optimizer: torch.optim.Optimizer,
    cost_function,
    epoch: int,
    title: str,
    device: str = "cuda",
) -> Tuple[float, float]:
    r""""""
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
        loss = cost_function(train_output[:, train.to_eval], label[:, train.to_eval])
        cumulative_loss += loss.item()

        predicted = constr_output.data > 0.5

        # fetch prediction and loss value
        total_train += inputs.shape[0] * inputs.shape[1]

        # compute training accuracy
        cumulative_accuracy += (predicted == label.byte()).sum().item()

        # backward pass
        loss.backward()
        # optimizer
        optimizer.step()

    return cumulative_loss / len(train_loader), cumulative_accuracy / total_train
