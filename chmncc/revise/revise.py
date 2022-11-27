import torch
import torch.nn as nn
from chmncc.utils import dotdict
from typing import Tuple
from chmncc.utils import get_constr_out
from typing import Union
from chmncc.loss import RRRLoss, IGRRRLoss


def revise_step(
    net: nn.Module,
    training_samples: torch.Tensor,
    ground_truths: torch.Tensor,
    confunder_masks: torch.Tensor,
    train: dotdict,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    revive_function: Union[IGRRRLoss, RRRLoss],
    device: str = "cuda",
) -> Tuple[float, float, float, float]:
    """Revise step of the network. It integrates the user feedback and revise the network by the means
    of the RRRLoss.

    Args:
        net [nn.Module] network on device
        training_samples [torch.Tensor]: training samples stacked tensor
        ground_truths [torch.Tensor]: groundtruths samples stacked tensor
        confounder_mask [torch.Tensor]: confounder masks stacked tensor
        train [dotdict]: training set dictionary
        R [torch.Tensor]: adjency matrix
        optimizer [torch.optim.Optimizer]: optimizer
        cost_function [Union[IGRRRLoss, RRRLoss]] RRR loss flavour
        device [str]: on which device to run the experiment [default: cuda]

    Returns:
        loss [float]
        right_answer_loss [float]: error considering the training loss
        right_reason_loss [float]: error considering the penalty on the wrong focus of the model
        accuracy [float]: accuracy of the model in percentage
    """
    # set the network to training mode
    net.train()

    # load data into device
    sample = training_samples.to(device)
    # ground_truth element
    ground_truth = ground_truths.to(device)
    # confounder mask
    confounder_mask = confunder_masks.to(device)

    # gradients reset
    optimizer.zero_grad()

    # output
    outputs = net(sample.float())

    # general prediction loss computation
    # MCLoss (their loss)
    constr_output = get_constr_out(outputs, R)
    train_output = ground_truth * outputs.double()
    train_output = get_constr_out(train_output, R)
    train_output = (
        1 - ground_truth
    ) * constr_output.double() + ground_truth * train_output

    # get the loss masking the prediction on the root -> confunder
    loss, right_answer_loss, right_reason_loss = revive_function(
        X=sample,
        y=ground_truth[:, train.to_eval],
        expl=confounder_mask,
        logits=train_output[:, train.to_eval],
    )

    predicted = constr_output.data > 0.5

    # compute training accuracy
    accuracy = (predicted == ground_truth.byte()).sum().item()

    # for calculating loss, acc per epoch
    right_answer_loss = right_answer_loss.item()
    right_reason_loss = right_reason_loss.item()

    # backward pass
    loss.backward()
    # optimizer
    optimizer.step()

    return (
        loss.item(),
        right_answer_loss,
        right_reason_loss,
        100 * accuracy / (ground_truth.shape[0] * ground_truth.shape[1]),
    )
