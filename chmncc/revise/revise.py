import torch
import tqdm
import torch.nn as nn
from chmncc.utils import dotdict
from typing import Tuple
from chmncc.utils import get_constr_out
from typing import Union
from chmncc.loss import RRRLoss, IGRRRLoss
from sklearn.metrics import average_precision_score


def revise_step(
    net: nn.Module,
    debug_loader: torch.utils.data.DataLoader,
    train: dotdict,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    revive_function: Union[IGRRRLoss, RRRLoss],
    title: str,
    batches_treshold: float,
    device: str = "cuda",
    have_to_train: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """Revise step of the network. It integrates the user feedback and revise the network by the means
    of the RRRLoss.

    Args:
        net [nn.Module] network on device
        debug_loader [torch.utils.data.DataLoader]: debug loader
        train [dotdict]: training set dictionary
        R [torch.Tensor]: adjency matrix
        optimizer [torch.optim.Optimizer]: optimizer
        revive_function [Union[IGRRRLoss, RRRLoss]]: revive function (RRR loss)
        title [str]: title for tqdm
        batches_treshold [float]: threshold for the batches
        device [str]: on which device to run the experiment [default: cuda]
        have_to_train [bool]: whether to train or not the model

    Returns:
        loss [float]
        right_answer_loss [float]: error considering the training loss
        right_reason_loss [float]: error considering the penalty on the wrong focus of the model
        accuracy [float]: accuracy of the model in percentage
        score [float]: area under the precision/recall curve
        right_reason_loss_confounded [float]: cumulative right_reason_loss divided over the number of confounded samples
    """
    total_train = 0.0
    comulative_loss = 0.0
    cumulative_accuracy = 0.0
    cumulative_right_answer_loss = 0.0
    cumulative_right_reason_loss = 0.0
    confounded_samples = 0.0

    # set the network to training mode
    if have_to_train:
        net.train()
    else:
        net.eval()

    # iterate over the training set
    for batch_idx, inputs in tqdm.tqdm(
        enumerate(debug_loader),
        desc=title,
    ):
        # get items
        (sample, ground_truth, confounder_mask, confounded, _, _) = inputs

        # load data into device
        sample = sample.to(device)
        sample.requires_grad = True
        # ground_truth element
        ground_truth = ground_truth.to(device)
        # confounder mask
        confounder_mask = confounder_mask.to(device)
        # confounded
        confounded = confounded.to(device)

        # gradients reset
        if have_to_train:
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

        # compute the amount of confounded samples
        confounded_samples += confounded.sum().item()

        predicted = constr_output.data > 0.5

        # fetch prediction and loss value
        total_train += ground_truth.shape[0] * ground_truth.shape[1]

        # compute training accuracy
        cumulative_accuracy += (predicted == ground_truth.byte()).sum().item()

        # for calculating loss, acc per epoch
        comulative_loss += loss.item()
        cumulative_right_answer_loss += right_answer_loss.item()
        cumulative_right_reason_loss += right_reason_loss.item()

        if have_to_train:
            # backward pass
            loss.backward()
            # optimizer
            optimizer.step()

        # compute the au(prc)
        predicted = predicted.to("cpu")
        cpu_constrained_output = train_output.to("cpu")
        ground_truth = ground_truth.to("cpu")

        if batch_idx == 0:
            predicted_train = predicted
            constr_train = cpu_constrained_output
            y_test = ground_truth
        else:
            predicted_train = torch.cat((predicted_train, predicted), dim=0)
            constr_train = torch.cat((constr_train, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, ground_truth), dim=0)

        # break if the number of training batches is more then the threshold
        if batch_idx >= batches_treshold:
            break

    # average precision score
    score = average_precision_score(
        y_test[:, train.to_eval], constr_train.data[:, train.to_eval], average="micro"
    )

    # confounded samples
    if confounded_samples:
        confounded_samples = 1

    return (
        comulative_loss / len(debug_loader),
        cumulative_right_answer_loss / len(debug_loader),
        cumulative_right_reason_loss / len(debug_loader),
        cumulative_accuracy / total_train * 100,
        score,
        cumulative_right_reason_loss / confounded_samples,
    )
