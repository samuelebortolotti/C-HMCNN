import torch
import tqdm
import torch.nn as nn
import numpy as np
from chmncc.utils import dotdict
from typing import Tuple
from chmncc.utils import get_constr_out, force_prediction_from_batch
from typing import Union
from chmncc.loss import RRRLoss, IGRRRLoss
from sklearn.metrics import average_precision_score
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from torchviz import make_dot


def show_computational_graph(
    net: nn.Module,
    output,
    folder_where_to_save: str,
    prefix: str,
    show_attrs=False,
    show_saved=False,
) -> None:
    """Shows the computational graph though the network by means of the graphviz library and saves the figure

    Args:
        net [nn.Module] network
        output: the output of the network
        folder_where_to_save: folder where to save the image
        prefix [str]: prefix of the image path
        show_attrs [False] show attr option for make_dot
        show_saved [False] show saved option for make_dot
    """
    graphviz = make_dot(
        output.mean(),
        params=dict(net.named_parameters()),
        show_attrs=show_attrs,
        show_saved=show_saved,
    )
    graphviz.render(
        "dot",
        outfile="{}/{}_computational_graph.pdf".format(folder_where_to_save, prefix),
    )


def show_gradient_behavior(
    named_parameters, folder_where_to_save: str, prefix: str
) -> None:
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    See: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8
    Credits: @RoshanRane

    Args:
        - folder_where_to_save [str]: where to save the plots
        - prefix [str]: image name prefix
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig = plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    fig.savefig(
        "{}/{}_gradient_analysis".format(folder_where_to_save, prefix), dpi=fig.dpi
    )
    plt.close()


def revise_step(
    epoch_number: int,
    net: nn.Module,
    debug_loader: torch.utils.data.DataLoader,
    train: dotdict,
    R: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    revive_function: Union[IGRRRLoss, RRRLoss],
    title: str,
    batches_treshold: float,
    folder_where_to_save: str,
    device: str = "cuda",
    have_to_train: bool = True,
    gradient_analysis: bool = False,
    prediction_treshold: float = 0.5,
    force_prediction: bool = False,
) -> Tuple[float, float, float, float, float, float, float]:
    """Revise step of the network. It integrates the user feedback and revise the network by the means
    of the RRRLoss.

    Args:
        epoch_number [int]: epoch number
        net [nn.Module] network on device
        train [dotdict]: training set dictionary
        R [torch.Tensor]: adjency matrix
        optimizer [torch.optim.Optimizer]: optimizer
        revive_function [Union[IGRRRLoss, RRRLoss]]: revive function (RRR loss)
        title [str]: title for tqdm
        batches_treshold [float]: threshold for the batches
        device [str]: on which device to run the experiment [default: cuda]
        have_to_train [bool]: whether to train or not the model
        gradient_analysis [bool]: whether to analyze the gradients by means of plots
        prediction_treshold [float]: threshold used to consider a class as predicted
        force_prediction [bool]: force prediction

    Returns:
        loss [float]
        right_answer_loss [float]: error considering the training loss
        right_reason_loss [float]: error considering the penalty on the wrong focus of the model
        accuracy [float]: accuracy of the model in percentage
        score [float]: area under the precision/recall curve raw
        score [float]: area under the precision/recall curve const
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

    for batch_idx, inputs in tqdm.tqdm(
        enumerate(debug_loader),
        desc=title,
    ):

        (sample, ground_truth, confounder_mask, confounded, superc, subc) = inputs

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
            confounded=confounded,
        )

        # compute the amount of confounded samples
        confounded_samples += confounded.sum().item()

        if force_prediction:
            predicted = force_prediction_from_batch(
                constr_output.data, prediction_treshold, False
            )
        else:
            predicted = constr_output.data > prediction_treshold  # 0.5

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
            if gradient_analysis:
                show_computational_graph(
                    net=net,
                    output=outputs,
                    folder_where_to_save=folder_where_to_save,
                    prefix="{}_{}".format(epoch_number, batch_idx),
                )
                show_gradient_behavior(
                    net.named_parameters(),
                    folder_where_to_save,
                    prefix="{}_{}".format(epoch_number, batch_idx),
                )
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

    # average precision score
    score_raw = average_precision_score(
        y_test[:, train.to_eval], constr_train.data[:, train.to_eval], average="micro"
    )

    score_const = average_precision_score(
        y_test[:, train.to_eval],
        predicted_train.data[:, train.to_eval].to(torch.float),
        average="micro",
    )

    # confounded samples
    if confounded_samples == 0:
        confounded_samples = 1

    return (
        comulative_loss / len(debug_loader),
        cumulative_right_answer_loss / len(debug_loader),
        cumulative_right_reason_loss / len(debug_loader),
        cumulative_accuracy / total_train * 100,
        score_raw,
        score_const,
        cumulative_right_reason_loss / confounded_samples,
    )
