"""Train module"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from chmncc.utils import (
    dotdict,
    force_prediction_from_batch,
    cross_entropy_from_softmax,
)
import numpy as np
from typing import Tuple, Optional
import tqdm
from chmncc.utils import get_constr_out
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    average_precision_score,
    hamming_loss,
    jaccard_score,
)
import itertools

from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE


def training_step(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train: dotdict,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cost_function: torch.nn.modules.loss.BCELoss,
    title: str,
    device: str = "cuda",
    constrained_layer: bool = True,
    prediction_treshold: float = 0.5,
    force_prediction: bool = False,
    use_softmax: bool = False,
    superclasses_number: int = 20,
) -> Tuple[float, float, float, float, Optional[float], Optional[float]]:
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
        constrained_layer [bool]: whether to use the constrained layer training from Giuchiglia et al.
        prediction_treshold [float]: threshold used to consider a prediction
        force_prediction [bool]: force the prediction
        use_softmax [bool]: use the softmax
        superclasses_number [int]: number of superclasses

    Returns:
        cumulative loss [float]
        accuracy [float] in percentange
        au prc raw [float]
        au prc const [float]
        rigth_answer_parent [Optional[float]] right answer for the parent
        rigth_answer_children [Optional[float]] right answer for the children
    """
    total_train = 0.0
    cumulative_loss = 0.0
    cumulative_loss_parent = None
    cumulative_loss_children = None
    cumulative_accuracy = 0.0

    # set the network to training mode
    net.train()

    # iterate over the training set
    for batch_idx, inputs in tqdm.tqdm(
        enumerate(itertools.islice(train_loader, 1, 5000)), desc=title
    ):

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
        if constrained_layer:
            # MCLoss (their loss)
            constr_output = get_constr_out(outputs, R)
            train_output = label * outputs.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1 - label) * constr_output.double() + label * train_output
        else:
            # fake the constrained output and training output
            constr_output = outputs
            train_output = label * outputs.double()
            train_output = (1 - label) * constr_output.double() + label * train_output

        if use_softmax:
            loss, loss_parent, loss_children = cross_entropy_from_softmax(
                label, train_output, superclasses_number
            )
        else:
            loss = cost_function(
                train_output[:, train.to_eval], label[:, train.to_eval]
            )
            # set them to None
            loss_parent, loss_children = None, None

        cumulative_loss += loss.item()

        # sum up the losses
        if loss_parent is not None and loss_children is not None:
            if cumulative_loss_children is None or cumulative_loss_parent is None:
                cumulative_loss_children = 0
                cumulative_loss_parent = 0
            cumulative_loss_parent += loss_parent.item()
            cumulative_loss_children += loss_children.item()

        # force prediction
        if force_prediction:
            predicted = force_prediction_from_batch(
                constr_output.data,
                prediction_treshold,
                use_softmax,
                superclasses_number,
            )
        else:
            predicted = constr_output.data > prediction_treshold  # 0.5

        # fetch prediction and loss value
        total_train += label.shape[0] * label.shape[1]

        # compute training accuracy
        cumulative_accuracy += (predicted == label.byte()).sum().item()

        # backward pass
        loss.backward()
        net.float()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        # optimizer
        optimizer.step()

        # compute the au(prc)
        predicted = predicted.to("cpu")
        cpu_constrained_output = train_output.to("cpu")
        label = label.to("cpu")

        if batch_idx == 0:
            predicted_train = predicted
            constr_train = cpu_constrained_output
            y_test = label
        else:
            predicted_train = torch.cat((predicted_train, predicted), dim=0)
            constr_train = torch.cat((constr_train, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, label), dim=0)

        # TODO force exit
        if batch_idx == 200:
            break

    # average precision score
    score_raw = average_precision_score(
        y_test[:, train.to_eval], constr_train.data[:, train.to_eval], average="micro"
    )

    # average precision score
    score_const = average_precision_score(
        y_test[:, train.to_eval],
        predicted_train.data[:, train.to_eval].to(torch.float),
        average="micro",
    )

    return (
        cumulative_loss / len(train_loader),
        cumulative_accuracy / total_train * 100,
        score_raw,
        score_const,
        None
        if cumulative_loss_parent is None
        else cumulative_loss_parent / len(train_loader),
        None
        if cumulative_loss_children is None
        else cumulative_loss_children / len(train_loader),
    )


def training_step_with_gate(
    net: nn.Module,
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    train: dotdict,
    title: str,
    device: str = "cuda",
) -> Tuple[float, float, float, float, float]:
    net.train()
    gate.train()

    total_train = 0
    tot_loss = 0
    tot_accuracy = 0

    for batch_idx, inputs in tqdm.tqdm(
        enumerate(itertools.islice(train_loader, 1, 5000)), desc=title
    ):
        # according to the Giunchiglia dataset
        inputs, labels = inputs

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # MCLoss
        output = net(inputs.float())
        thetas = gate(output)
        cmpe.set_params(thetas)

        loss = cmpe.cross_entropy(labels, log_space=True).mean()
        # predicted
        cmpe.set_params(thetas)
        predicted = (cmpe.get_mpe_inst(inputs.shape[0]) > 0).long()

        # fetch prediction and loss value
        total_train += labels.shape[0] * labels.shape[1]

        # compute training accuracy
        tot_accuracy += (predicted == labels.byte()).sum().item()

        tot_loss += loss
        loss.backward()
        optimizer.step()

        # compute the au(prc)
        predicted = predicted.to("cpu")
        labels = labels.to("cpu")

        if batch_idx == 0:
            predicted_train = predicted
            y_test = labels
        else:
            predicted_train = torch.cat((predicted_train, predicted), dim=0)
            y_test = torch.cat((y_test, labels), dim=0)

        # TODO increase
        #  if batch_idx == 200:
        #      break

    y_test = y_test[:, train.to_eval]
    predicted_train = predicted_train.data[:, train.to_eval].to(torch.float)

    # jaccard score
    jaccard = jaccard_score(y_test, predicted_train, average="micro")
    # hamming score
    hamming = hamming_loss(y_test, predicted_train)
    # average precision score
    auprc_score = average_precision_score(y_test, predicted_train, average="micro")
    # accuracy
    accuracy = tot_accuracy / total_train

    return tot_loss / total_train, accuracy * 100, jaccard, hamming, auprc_score
