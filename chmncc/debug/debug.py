"""Model which provides the debugging facility to the model"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from numpy.lib.function_base import iterable
import torch
import os
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from chmncc.networks import ResNet18, LeNet5, LeNet7, AlexNet, MLP
from chmncc.optimizers.adam import get_adam_optimizer_with_gate
from chmncc.utils.utils import (
    force_prediction_from_batch,
    load_last_weights,
    load_last_weights_gate,
    log_value,
    load_best_weights,
    load_best_weights_gate,
    grouped_boxplot,
    prediction_statistics_boxplot,
    plot_confusion_matrix_statistics,
    plot_global_multiLabel_confusion_matrix,
    get_lr,
    get_confounders_and_hierarchy,
    prepare_dict_label_predictions_from_raw_predictions,
    plot_confounded_labels_predictions,
    prepare_empty_probabilistic_circuit,
    prepare_probabilistic_circuit
)
from chmncc.dataset import (
    load_dataloaders,
    get_named_label_predictions,
    get_hierarchical_index_from_named_label,
    LoadDebugDataset
)
from chmncc.config import label_confounders
import wandb
from chmncc.test import test_step, test_step_with_prediction_statistics, test_circuit, test_step_with_prediction_statistics_with_gates
from chmncc.explanations import compute_integrated_gradient, output_gradients
from chmncc.loss import RRRLoss, IGRRRLoss, RRRLossWithGate
from chmncc.revise import revise_step, revise_step_with_gates
from chmncc.optimizers import get_adam_optimizer, get_plateau_scheduler, get_sdg_optimizer_with_gate
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Tuple, Union, List
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats
from sklearn.linear_model import RidgeClassifier
from torchsummary import summary
from itertools import tee
import itertools

from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE

def save_sample(
    dataset: str,
    train_sample: torch.Tensor,
    prediction: torch.Tensor,
    idx: int,
    debug_folder: str,
    dataloaders: Dict[str, Any],
    superclass: str,
    subclass: str,
    integrated_gradients: bool,
    prefix: str,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    use_circuits: bool,
    gate: DenseGatingFunction = None,
    cmpe: CircuitMPE = None,
) -> Tuple[bool, np.ndarray]:
    """Save the test sample.
    Then it returns whether the sample contains a confunder or if the sample has been
    guessed correctly.

    Args:
        dataset: [str]: dataset used
        train_sample [torch.Tensor]: train sample depicting the image
        prediction [torch.Tensor]: prediction of the network on top of the test function
        idx [int]: index of the element of the batch to consider
        debug_folder [str]: string depicting the folder where to save the figures
        dataloaders [Dict[str, Any]]: dictionary depicting the dataloaders of the data
        superclass [str]: superclass string
        subclass [str]: subclass string
        integrated_gradients [bool]: whether to use integrated gradients
        prefix [str]: prefix for the sample to save
        prediction_treshold [float] prediction threshold
        force_prediction [bool]: force prediction
        use_softmax [bool]: use softmax
        use_circuits [bool]: use probabilistic circuits
        gate [DenseGatingFunction]: gating function
        cmpe [CircuitMPE]: circuit = None

    Returns:
        correct_guess [bool]: whether the model has guessed correctly
        single_el_show [np.ndarray]: single element
    """
    # prepare the element in order to show it
    single_el_show = train_sample.squeeze(0).clone().cpu().data.numpy()
    single_el_show = single_el_show.transpose(1, 2, 0)

    # normalize
    single_el_show = np.fabs(single_el_show)
    single_el_show = single_el_show / np.max(single_el_show)

    # get the prediction
    if not use_circuits:
        if force_prediction:
            predicted_1_0 = force_prediction_from_batch(prediction.cpu().data, prediction_treshold, use_softmax)
        else:
            predicted_1_0 = prediction.cpu().data > prediction_treshold
    else:
        # prediction out of the circuit
        predicted_1_0 = prediction.cpu()

    predicted_1_0 = predicted_1_0.to(torch.float)[0]
    # get the named prediction
    named_prediction = get_named_label_predictions(
        predicted_1_0, dataloaders["test_set"].get_nodes()
    )

    # extract parent and children prediction
    _, children_values, parents = get_confounders_and_hierarchy(dataset)

    children = [
        element for element_list in children_values for element in element_list
    ]
    parent_predictions = list(filter(lambda x: x in parents, named_prediction))
    children_predictions = list(filter(lambda x: x in children, named_prediction))

    # check whether it is confunded
    correct_guess = False

    # set the guess as correct if it is
    if len(parent_predictions) == 1 and len(children_predictions) == 1:
        if (
            superclass.strip() == parent_predictions[0].strip()
            and subclass == children_predictions[0].strip()
        ):
            correct_guess = True

    # plot the title
    prediction_text = "Predicted: {}\nbecause of: {}".format(
        parent_predictions, children_predictions
    )

    # get figure
    fig = plt.figure()
    if single_el_show.shape[2] == 1:
        plt.imshow(single_el_show, cmap="gray")
    else:
        plt.imshow(single_el_show)
    plt.title(
        "Groundtruth superclass: {} \nGroundtruth subclass: {}\n\n{}".format(
            superclass, subclass, prediction_text
        )
    )
    plt.tight_layout()
    # show the figure
    fig.savefig(
        "{}/{}_iter_{}_original{}{}.pdf".format(
            debug_folder,
            prefix,
            idx,
            "_integrated_gradients" if integrated_gradients else "_input_gradients",
            "_correct" if correct_guess else "",
        ),
        dpi=fig.dpi,
    )
    # close the figure
    plt.close(fig)

    # whether the prediction was right
    return correct_guess, single_el_show


def visualize_sample(
    dataset: str,
    single_el: torch.Tensor,
    debug_folder: str,
    idx: int,
    integrated_gradients: bool,
    superclass: str,
    subclass: str,
    dataloaders: Dict[str, Any],
    device: str,
    net: nn.Module,
    prefix: str,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    use_circuits: bool,
    gate: DenseGatingFunction = None,
    cmpe: CircuitMPE = None,
) -> None:
    """Save the samples information, including the sample itself, the gradients and the masked gradients

    Args:
        dataset [str]: which dataset is employed
        single_el [torch.Tensor]: sample to show
        debug_folder [str]: folder where to store the data
        idx [int]: idex of the element to save
        integrated_gradients [bool]: whether to use integrated gradiends
        superclass [str]: superclass of the sample
        subclass [str]: subclass of the sample
        dataloaders [Dict[str, Any]]: dataloaders
        device [str]: device
        net [nn.Module]: network
        prefix [str]: prefix for the sample to save
        prediction_treshold [float]: prediction treshold
        force_prediction [bool]: force prediction
        use_softmax [bool]: use softmax
        use_circuits [bool]: use probabilistic circuits
        gate [DenseGatingFunction]: gating function
        cmpe [CircuitMPE]: circuit = None
    """
    # set the network to eval mode
    net.eval()
    if use_circuits and gate is not None:
        gate.eval()

    # unsqueeze the element
    single_el = torch.unsqueeze(single_el, 0)

    # set the gradients as required
    single_el.requires_grad = True

    # move the element to device
    single_el = single_el.to(device)

    # get the predictions
    output = net(single_el.float())
    if use_circuits and gate is not None:
        thetas = gate(output.float())
        cmpe.set_params(thetas)
        preds = (cmpe.get_mpe_inst(single_el.shape[0]) > 0).long()
    else:
        preds = output

    # save the sample and whether the sample has been correctly guessed
    correct_guess, sample_to_save = save_sample(
        dataset=dataset,
        train_sample=single_el,
        prediction=preds,
        idx=idx,
        debug_folder=debug_folder,
        dataloaders=dataloaders,
        superclass=superclass,
        subclass=subclass,
        integrated_gradients=integrated_gradients,
        prefix=prefix,
        prediction_treshold=prediction_treshold,
        force_prediction=force_prediction,
        use_softmax=use_softmax,
        use_circuits=use_circuits,
        gate=gate,
        cmpe=cmpe,
    )

    # compute the gradient, whether input or integrated
    gradient = compute_gradients(
        single_el=single_el,
        net=net,
        integrated_gradients=integrated_gradients,
    )

    # show the gradient
    gradient_to_show, max_value = show_gradient(
        gradient=gradient,
        debug_folder=debug_folder,
        idx=idx,
        correct_guess=correct_guess,
        integrated_gradients=integrated_gradients,
        prefix=prefix,
    )

    overlay_input_gradient(
        gradient_to_show=gradient_to_show,
        single_el=sample_to_save,
        max_value=max_value,
        debug_folder=debug_folder,
        idx=idx,
        integrated_gradients=integrated_gradients,
        correct_guess=correct_guess,
        full=True,
        prefix=prefix
    )


def show_masked_gradient(
    confounder_mask: torch.Tensor,
    gradient: torch.Tensor,
    debug_folder: str,
    idx: int,
    integrated_gradients: bool,
    correct_guess: bool,
    prefix: str,
) -> Tuple[np.ndarray, float]:
    """Save the masked gradient

    Args:
        confounder_mask [torch.Tensor]: confounder mask
        gradient [torch.Tensor]: gradient to save, either the input or integrated
        debug_folder [str]: where to save the sample
        idx [int]: idex of the element to save
        integrated_gradients [bool]: whether to use integrated gradiends
        correct_guess [bool]: whether the sample has been guessed correctly
        prefix [str]: prefix of the image name to be saved

    Returns:
        gradient_to_show, np.max(gradient_to_show_absolute_values) [Tuple[np.ndarray, float]], gradient and gradient max value
    """

    # get the gradient
    gradient_to_show = gradient.clone().cpu().data.numpy()
    gradient_to_show_absolute_values = np.fabs(gradient_to_show)
    gradient_to_show_absolute_values_masked = np.where(confounder_mask > 0.5, gradient_to_show_absolute_values, 0)
    # normalize the value
    gradient_to_show = gradient_to_show_absolute_values_masked / np.max(
        gradient_to_show_absolute_values
    )
    # norm color
    norm = matplotlib.colors.Normalize(
        vmin=0, vmax=np.max(gradient_to_show_absolute_values)
    )

    # show the picture
    fig = plt.figure()
    plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="gray"), label="Gradient magnitude"
    )
    plt.imshow(gradient_to_show, cmap="gray")
    plt.title(
        "{} gradient only confunder zone".format(
            "Integrated" if integrated_gradients else "Input"
        )
    )
    # show the figure
    fig.savefig(
        "{}/{}_iter_{}_gradient_only_confunder_{}{}.pdf".format(
            debug_folder,
            prefix,
            idx,
            "integrated" if integrated_gradients else "input",
            "_correct" if correct_guess else "",
        ),
        dpi=fig.dpi,
    )
    plt.close(fig)

    return gradient_to_show, np.max(gradient_to_show_absolute_values)

def overlay_input_gradient(
    gradient_to_show: torch.Tensor,
    single_el: torch.Tensor,
    max_value: float,
    debug_folder: str,
    idx: int,
    integrated_gradients: bool,
    correct_guess: bool,
    prefix: str,
    full: bool
) -> None:
    """Overlay the input gradient over the base grayscale image

    Args:
        gradient_to_show [torch.Tensor]: gradient to show
        single_el [torch.Tensor]: element to show
        max_value [float]: max value of the gradient
        debug_folder [str]: folder where to save the images
        idx [int]: index of the image
        integrated_gradients [bool]: whether we are asking to addd the integrated gradient
        prefix [str]: prefix of the image
        full [bool]: full
    """

    # norm color
    norm = matplotlib.colors.Normalize(
        vmin=0, vmax=max_value
    )
    # show the picture
    fig = plt.figure()
    plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis'), label="Gradient magnitude"
    )
    plt.imshow(single_el, cmap="gray")
    plt.imshow(gradient_to_show, cmap='viridis', alpha=0.5)
    plt.title("{} gradient overlay".format("Integrated" if integrated_gradients else "Input"))

    # show the figure
    fig.savefig(
        "{}/{}_iter_{}_overlayed_{}_image_{}{}.pdf".format(
            debug_folder,
            prefix,
            idx,
            "full" if full else "",
            "integrated" if integrated_gradients else "input",
            "_correct" if correct_guess else "",
        ),
        dpi=fig.dpi,
    )

    plt.close(fig)

def show_gradient(
    gradient: torch.Tensor,
    debug_folder: str,
    idx: int,
    correct_guess: bool,
    integrated_gradients: bool,
    prefix: str,
) -> Tuple[np.ndarray, float]:
    """Save the gradient

    Args:
        gradient [torch.Tensor]: gradient to save, either the input or integrated
        debug_folder [str]: where to save the sample
        idx [int]: idex of the element to save
        correct_guess [bool]: whether the sample has been guessed correctly
        integrated_gradients [bool]: whether to use integrated gradiends
        prefix [str]: prefix to save the pictures with

    Returns:
        gradient_to_show, np.max(gradient_to_show_absolute_values) [Tuple[np.ndarray, float]], gradient and gradient max value
    """
    # get the gradient
    gradient_to_show = gradient.clone().cpu().data.numpy()
    gradient_to_show_absolute_values = np.fabs(gradient_to_show)

    # normalize the value
    gradient_to_show = gradient_to_show_absolute_values / np.max(
        gradient_to_show_absolute_values
    )

    # norm color
    norm = matplotlib.colors.Normalize(
        vmin=0, vmax=np.max(gradient_to_show_absolute_values)
    )

    # show the picture
    fig = plt.figure()
    plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="gray"), label="Gradient magnitude"
    )
    plt.imshow(gradient_to_show, cmap="gray")
    plt.title("{} gradient".format("Integrated" if integrated_gradients else "Input"))

    # show the figure
    fig.savefig(
        "{}/{}_iter_{}_gradient_confunder_{}{}.pdf".format(
            debug_folder,
            prefix,
            idx,
            "integrated" if integrated_gradients else "input",
            "_correct" if correct_guess else "",
        ),
        dpi=fig.dpi,
    )
    plt.close(fig)

    return gradient_to_show, np.max(gradient_to_show_absolute_values)


def compute_gradients(
    single_el: torch.Tensor,
    net: nn.Module,
    integrated_gradients: bool,
) -> torch.Tensor:
    """Computes the integrated graditents or input gradient according to what the user specify in the argument

    Args:
        single_el [torch.Tensor]: sample depicting the image
        net [nn.Module]: neural network
        integrated_gradients [bool]: whether the method uses integrated gradients or input gradients
    Returns:
        integrated_gradient [torch.Tensor]: integrated_gradient
    """
    if integrated_gradients:
        # integrated gradients
        gradient = compute_integrated_gradient(
            single_el, torch.zeros_like(single_el), net
        )
    else:
        # integrated gradients
        gradient = output_gradients(single_el, net(single_el))[0]

    # sum over RGB channels
    gradient = torch.sum(gradient, dim=0)

    return gradient


def save_some_confounded_samples(
    dataset: str,
    net: nn.Module,
    start_from: int,
    number: int,
    loader: torch.utils.data.DataLoader,
    device: str,
    dataloaders: Dict[str, Any],
    folder: str,
    integrated_gradients: bool,
    prefix: str,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    use_circuits: bool,
    gate: DenseGatingFunction = None,
    cmpe: CircuitMPE = None,
) -> None:
    """Save some confounded examples according to the dataloader and the number of examples the user specifies

    Args:
        dataset [str]: which dataset to employ
        net [nn.Module]: neural network
        start_from [int]: start enumerating the sample from
        number [int]: stop enumerating the samples when the counter has reached (note that it has to start from start_from)
        loader [torch.utils.data.DataLoader]: loader where to save the samples
        device [str]: device
        dataloaders [Dict[str, Any]: dataloaders
        folder [str]: folder where to store the sample
        integrated_gradients [bool]: whether the gradient are integrated or input gradients
        prefix [str]: prefix to save the images with
        prediction_treshold [float] prediction treshold
        force_prediction [bool] force prediction
        use_softmax [bool] use softmax
        use_circuits [bool]: use circuits for evaluation
        gate [DenseGatingFunction] = None: gating function for the circuit
        cmpe [CircuitMPE] = None: circuit
    """
    # set the networ to evaluation mode
    net.eval()
    if use_circuits and gate is not None:
        gate.eval()

    # set the counter
    counter = start_from
    # set the done flag
    done = False
    for _, inputs in tqdm.tqdm(
        enumerate(loader),
        desc="Save",
    ):
        (sample, _, confounder_mask, confounded, superclass, subclass) = inputs
        # loop over the batch
        for i in range(sample.shape[0]):
            # is the element confound?
            if confounded[i]:
                # visualize it
                visualize_sample(
                    dataset,
                    sample[i],
                    folder,
                    counter,
                    integrated_gradients,
                    superclass[i],
                    subclass[i],
                    dataloaders,
                    device,
                    net,
                    prefix,
                    prediction_treshold,
                    force_prediction,
                    use_softmax,
                    use_circuits,
                    gate,
                    cmpe
                )
                # increase the counter
                counter += 1
                # stop when done
                if counter == number:
                    done = True
                    break
        # stop when done
        if done:
            break


def make_meshgrid(x, y, h=0.1, padding=1.0):
    """Make the Meshgrid

    taken from:
    Andrea Passerini's lab notes on sklearn
    https://disi.unitn.it/~passerini/teaching/2021-2022/MachineLearning/index.html
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - padding, y.max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot contours of the plot

    taken from:
    Andrea Passerini's lab notes on sklearn
    https://disi.unitn.it/~passerini/teaching/2021-2022/MachineLearning/index.html
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_decision_surface(X, Y, clf, x_label, y_label, title, jitter, h=0.1, padding=1.0, debug_mode=True):
    """Print the decision surface of a trained sklearn classifier

    taken from:
    Andrea Passerini's lab notes on sklearn
    https://disi.unitn.it/~passerini/teaching/2021-2022/MachineLearning/index.html
    """

    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, h, padding)

    colors = ["orange" if y == 1 else "blue" for y in Y]

    if jitter:
        for i in range(X0.shape[0]):
            X0[i] += np.random.uniform(low=-0.4, high=0.4)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)
    ax.scatter(X0, X1, c=colors, s=20)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend()

    custom = [
        matplotlib.lines.Line2D(
            [], [], marker=".", markersize=20, color="orange", linestyle="None"
        ),
        matplotlib.lines.Line2D(
            [], [], marker=".", markersize=20, color="blue", linestyle="None"
        ),
    ]

    if debug_mode:
        plt.legend(custom, ["Confounded (1)", "Not confounded (2)"], fontsize=10)
    else:
        plt.legend(custom, ["Correct [not conf] (1)", "Wrong [conf] (2)"], fontsize=10)
    plt.close(fig)
    return fig


def compute_gradient_confound_correlation(
    net: nn.Module,
    sample_dataloader: torch.utils.data.DataLoader,
    integrated_gradients: bool,
    sample_each: int,
    folder_where_to_save: str,
    figure_prefix_name: str,
    device: str,
) -> None:
    """Function which computes the gradient magnitude (computed with the standard L2 norm)
    and the fact that an example is confounded or not. To do that, it samples randomly
    a given number of samples of both counfounded and non confounded samples from the given
    dataloader.

    This, produces:
    - boxplot
    - pearson correlation and jittered plot with a standard linear classifier

    Args:
        net [nn.Module]: neural network
        sample_dataloader [torch.utils.data.DataLoader]: loader where to sample the samples
        integrated_gradients [bool]: whether to use integrated graidents or input gradients
        sample_each [int]: how many samples to sample from each group (confounded and not)
        folder_where_to_save [str]: where to save the plots produced
        figure_prefix_name [str]: prefix name which is used to save the data
        device [str]: device to use
    """
    # sequence of counfounded samples
    counfounded_sequence = []
    gradients_magnitude_sequence = []
    current_samples_number = [0, 0]
    have_to_redo = True
    emergency_escape = 10
    redone_iter = 0

    # network in evaluation mode
    net.eval()

    while have_to_redo:
        # loop over the samples
        for _, inputs in tqdm.tqdm(
            enumerate(sample_dataloader),
            desc="Gradient Magnitude Correlation",
        ):
            # get items
            (sample, _, _, confounded, _, _) = inputs
            bool_item = True
            confounder_index = 0

            # loop over the elements
            for element_idx in range(sample.shape[0]):
                if confounded[element_idx]:
                    bool_item = 1.0
                    confounder_index = 0
                else:
                    bool_item = 0.0
                    confounder_index = 1

                # already got the correct number of samples
                if current_samples_number[confounder_index] >= sample_each:
                    continue

                # increase the number of samples
                current_samples_number[confounder_index] += 1

                # prepare the sample
                single_el = torch.unsqueeze(sample[element_idx], 0)
                single_el.requires_grad = True
                single_el = single_el.to(device)
                # compute the gradient
                gradient = compute_gradients(
                    single_el=single_el,
                    net=net,
                    integrated_gradients=integrated_gradients,
                )

                gradient = gradient.to("cpu")

                # append elements
                gradients_magnitude_sequence.append(
                    torch.linalg.norm(torch.flatten(gradient), dim=0, ord=2)
                    .detach()
                    .numpy()
                    .item(),
                )
                counfounded_sequence.append(bool_item)

            # break if the number of examples is sufficient
            if sum(current_samples_number) >= 2 * sample_each:
                have_to_redo = False
                break

        # increase the redo
        redone_iter += 1

        if redone_iter >= emergency_escape:
            print("Escape a probable infinite loop")
            break

    # compute the correlation
    corr = scipy.stats.pearsonr(gradients_magnitude_sequence, counfounded_sequence)

    # stacking the gradients with the x position => 0
    gradients_magnitude_sequence = np.array(gradients_magnitude_sequence).reshape(-1, 1)
    gradients_magnitude_sequence_filled_x = np.column_stack(
        (
            np.zeros(gradients_magnitude_sequence.shape[0]),
            gradients_magnitude_sequence,
        )
    )
    counfounded_sequence = np.array(counfounded_sequence)

    # use a simple ridge classifier in order to binary separate the two classes
    rc = RidgeClassifier()

    rc.fit(gradients_magnitude_sequence_filled_x, counfounded_sequence)
    score = rc.score(gradients_magnitude_sequence_filled_x, counfounded_sequence)
    fig = plot_decision_surface(
        gradients_magnitude_sequence_filled_x,
        counfounded_sequence,
        rc,
        "x",
        "gradient_magnitude",
        "Score: {}, pearson correlation {:.3f}, p-val {:.3f}\nsignificant with 95% confidence {} #conf {} #not-conf {}".format(
            score,
            corr[0],
            corr[1],
            corr[1] < 0.05,
            current_samples_number[1],
            current_samples_number[0],
        ),
        True,
    )
    fig.savefig(
        "{}/{}_{}_gradient_correlation.pdf".format(
            folder_where_to_save,
            "ingegrated" if integrated_gradients else "input",
            figure_prefix_name,
        )
    )

    # print the boxplot
    confounded_indexes = np.array((counfounded_sequence > 0.5), dtype=bool)
    fig = plt.figure(figsize=(7, 4))
    bp_dict = plt.boxplot(
        [
            gradients_magnitude_sequence[confounded_indexes].squeeze(),
            gradients_magnitude_sequence[~confounded_indexes].squeeze(),
        ]
    )
    plt.xticks([1, 2], ["Confounded", "Not Confounded"])

    for line in bp_dict["medians"]:
        # get position data for median line
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        plt.text(x, y, "%.5f" % y, horizontalalignment="center")

    fig.suptitle(
        "Boxplot: #conf {} #not-conf {}".format(
            current_samples_number[1], current_samples_number[0]
        )
    )
    fig.savefig(
        "{}/{}_{}_gradient_boxplot.pdf".format(
            folder_where_to_save,
            "ingegrated" if integrated_gradients else "input",
            figure_prefix_name,
        )
    )
    plt.close(fig)


def debug(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    debug_folder: str,
    iterations: int,
    cost_function: torch.nn.BCELoss,
    device: str,
    set_wandb: bool,
    integrated_gradients: bool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    debug_test_loader: torch.utils.data.DataLoader,
    batch_size: int,
    test_batch_size: int,
    reviseLoss: Union[RRRLoss, IGRRRLoss],
    model_folder: str,
    network: str,
    gradient_analysis: bool,
    num_workers: int,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    dataset: str,
    balance_subclasses: List[str],
    balance_weights: List[float],
    correct_by_duplicating_samples: bool,
    use_probabilistic_circuits: bool,
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    use_gate_output: bool,
    montecarlo: bool,
    writer: SummaryWriter,
    exp_name: str,
    **kwargs: Any
) -> None:
    """Method which performs the debug step by fine-tuning the network employing the right for the right reason loss.

    Args:
        net [nn.Module]: neural network
        dataloaders [Dict[str, Any]]: dataloaders
        debug_folder [str]: debug_folder
        iterations [int]: number of the iterations
        cost_function [torch.nn.BCELoss]: criterion for the classification loss
        device [str]: device
        set_wandb [bool]: set wandb up
        integrated gradients [bool]: whether to use integrated gradients or input gradients
        optimizer [torch.optim.Optimizer]: optimizer to employ
        scheduler [torch.optim.lr_scheduler._LRScheduler]: scheduler to employ
        batch_size [int]: size of the batch
        test_batch_size [int]: size of the batch in the test settings
        reviseLoss: Union[RRRLoss, IGRRRLoss]: loss for feeding the network some feedbacks
        model_folder [str]: folder where to fetch the models weights
        network [str]: name of the network
        gradient_analysis [bool]: whether to analyze the gradient behavior
        num_workers [int]: number of workers for dataloaders
        prediction_treshold [float]: prediction threshold
        force_prediction [bool]: force prediction
        use_softmax [bool]: use softmax
        dataset [str]: which dataset is used
        balance_subclasses List[str]: which subclasses to rebalance
        balance_weights: List[float]: which weights are associated to which subclasses
        correct_by_duplicating_samples [bool]: whether to correct the confounded network by duplicating the samples or by balancing the weights
        use_probabilistic_circuits [bool]: whether to use probabilistic circuits
        gate [DenseGatingFunction]: gate
        cmpe [CircuitMPE]: circuit MPE
        use_gate_output [bool]: whether to use the circuit output for computing the RRR loss
        montecarlo [bool]: whether to use a Montecarlo approach during testing
        writer [SummaryWriter]: summary writer
        exp_name [str]: name of the experiment for Tensorboard
        **kwargs [Any]: kwargs
    """
    print("Have to run for {} debug iterations...".format(iterations))

    ## ==================================================================
    # Load debug datasets: for training the data -> it has labels and confounders position information
    debug_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position"],
        balance_subclasses if correct_by_duplicating_samples else [],
        balance_weights if correct_by_duplicating_samples else []
    )

    print("Debug dataset statistics:")
    debug_train.train_set.print_stats()

    test_debug = LoadDebugDataset(
        dataloaders["test_dataset_with_labels_and_confunders_pos"],
    )

    # Dataloaders for the previous values
    debug_loader = torch.utils.data.DataLoader(
        debug_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_debug = torch.utils.data.DataLoader(
        test_debug, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    ## ==================================================================

    # test with only confounder (in test) loaders
    test_only_confounder = dataloaders[
        "test_loader_with_labels_and_confunders_pos_only"
    ]

    for_test_loader_test_only_confounder_wo_conf = torch.utils.data.DataLoader(
        dataloaders["test_dataset_with_labels_and_confunders_pos_only_without_confounders"], batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    for_test_loader_test_only_confounder_wo_conf_in_train_data = torch.utils.data.DataLoader(
        dataloaders["test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples"], batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    # only label confounders dataloader
    labels_only_test_loader = dataloaders["test_loader_only_label_confounders"]

    # copy the iterators for trying to analyze the same images
    print_iterator_before = iter(test_debug)
    print_iterator_before, print_iterator_after = tee(print_iterator_before)

    # save some training samples
    save_some_confounded_samples(
        dataset=dataset,
        net=net,
        start_from=0,
        number=30,
        dataloaders=dataloaders,
        device=device,
        folder=debug_folder,
        integrated_gradients=integrated_gradients,
        loader=print_iterator_before,
        prefix="before",
        prediction_treshold=prediction_treshold,
        force_prediction=force_prediction,
        use_softmax=use_softmax,
        use_circuits=use_probabilistic_circuits,
        gate=gate,
        cmpe=cmpe
    )

    # compute graident confounded correlation
    #  compute_gradient_confound_correlation(
    #      net, debug_loader, integrated_gradients, 100, debug_folder, "train", device
    #  )

    # best test score
    best_test_score = 0.0

    # running for the requested iterations
    for it in range(iterations):

        print("Start iteration number {}".format(it))
        print("-----------------------------------------------------")

        # training with RRRloss feedbacks
        if use_probabilistic_circuits:
            (
                train_total_loss,
                train_total_right_answer_loss,
                train_total_right_reason_loss,
                train_right_reason_loss_confounded,
                train_total_accuracy,
                train_total_score_raw,
                train_hamming,
                train_jaccard,
            ) = revise_step_with_gates(
                net=net,
                gate=gate,
                cmpe=cmpe,
                debug_loader=iter(debug_loader),
                R=dataloaders["train_R"],
                train=dataloaders["train"],
                optimizer=optimizer,
                revive_function=reviseLoss,
                device=device,
                title="Train with RRR",
                use_gate_output=use_gate_output,
            )
        else:
            (
                train_total_loss,
                train_total_right_answer_loss,
                train_total_right_reason_loss,
                train_total_accuracy,
                train_total_score_raw,
                train_total_score_const,
                train_right_reason_loss_confounded,
                train_loss_parent,
                train_loss_children
            ) = revise_step(
                epoch_number=it,
                net=net,
                debug_loader=iter(debug_loader),
                R=dataloaders["train_R"],
                train=dataloaders["train"],
                optimizer=optimizer,
                revive_function=reviseLoss,
                device=device,
                title="Train with RRR",
                gradient_analysis=gradient_analysis,
                folder_where_to_save=debug_folder,
                prediction_treshold=prediction_treshold,
                force_prediction=force_prediction,
                use_softmax=use_softmax,
                superclasses_number=dataloaders["train_set"].n_superclasses,
            )

        if use_probabilistic_circuits:
            print(
                "\n\t Debug full loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, AuPRc raw {:.5f}, Hamming score {:.5f}, Jaccard score {:.5f}, Right Reason Loss on Confounded {:.5f}".format(
                    train_total_loss,
                    train_total_right_answer_loss,
                    train_total_right_reason_loss,
                    train_total_accuracy,
                    train_total_score_raw,
                    train_hamming,
                    train_jaccard,
                    train_right_reason_loss_confounded,
                )
            )
        else:
            print(
                "\n\t Debug full loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, AuPRc raw {:.5f}, AuPRc const {:.5f}, Right Reason Loss on Confounded {:.5f}".format(
                    train_total_loss,
                    train_total_right_answer_loss,
                    train_total_right_reason_loss,
                    train_total_accuracy,
                    train_total_score_raw,
                    train_total_score_const,
                    train_right_reason_loss_confounded,
                )
            )

        print("Testing...")

        # validation set
        if use_probabilistic_circuits:
            (
                val_loss,
                val_accuracy,
                val_jaccard,
                val_hamming,
                val_score_raw,
            ) = test_circuit(
                net=net,
                gate=gate,
                cmpe=cmpe,
                test_loader=iter(debug_test_loader),
                title="Validation",
                test=dataloaders["train"],
                device=device,
                montecarlo=montecarlo,
            )
        else:
            val_loss, val_accuracy, val_score_raw, val_score_const, val_loss_parent, val_loss_children = test_step(
                net=net,
                test_loader=iter(debug_test_loader),
                cost_function=cost_function,
                title="Validation",
                test=dataloaders["train"],
                device=device,
                prediction_treshold=prediction_treshold,
                force_prediction=force_prediction,
                superclasses_number=dataloaders["train_set"].n_superclasses,
                montecarlo=montecarlo,
            )

        if use_probabilistic_circuits:
            print(
                "\t [Validation set]: Loss {:.5f}, Validation accuracy {:.2f}%, Validation Jaccard Score {:.3f}, Validation Hamming Loss {:.3f}, Validation Area under Precision-Recall Curve Raw {:.3f}".format(
                    val_loss, val_accuracy, val_jaccard, val_hamming, val_score_raw
                )
            )
        else:
            print(
                "\n\t [Validation set]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve Raw {:.3f}, Area under Precision-Recall Curve Const {:.3f}".format(
                    val_loss, val_accuracy, val_score_raw, val_score_const
                )
            )


        if use_probabilistic_circuits:
            (
                test_loss_original,
                test_total_right_answer_loss,
                test_total_right_reason_loss,
                test_right_reason_loss_confounded,
                test_accuracy_original,
                test_score_original_raw,
                test_hamming,
                test_jaccard,
            ) = revise_step_with_gates(
                net=net,
                gate=gate,
                cmpe=cmpe,
                debug_loader=iter(test_debug),
                R=dataloaders["train_R"],
                train=dataloaders["test"],
                optimizer=optimizer,
                revive_function=reviseLoss,
                device=device,
                title="Test with RRR",
                have_to_train=False,
                use_gate_output=use_gate_output,
            )
        else:
            (
                test_loss_original,
                test_total_right_answer_loss,
                test_total_right_reason_loss,
                test_accuracy_original,
                test_score_original_raw,
                test_score_original_const,
                test_right_reason_loss_confounded,
                test_loss_parent,
                test_loss_children
            ) = revise_step(
                epoch_number=it,
                net=net,
                debug_loader=iter(test_debug),
                R=dataloaders["train_R"],
                train=dataloaders["test"],
                optimizer=optimizer,
                revive_function=reviseLoss,
                device=device,
                title="Test with RRR",
                gradient_analysis=gradient_analysis,
                folder_where_to_save=debug_folder,
                have_to_train=False,
                prediction_treshold=prediction_treshold,
                force_prediction=force_prediction,
                use_softmax=use_softmax,
                superclasses_number=dataloaders["train_set"].n_superclasses,
            )


        if use_probabilistic_circuits:
            print(
                "\n\t [Test set] Test loss {:.5f}, Test accuracy {:.2f}%, Test Jaccard Score {:.3f}, Test Hamming Loss {:.3f}, Test Area under Precision-Recall Curve Raw {:.3f}".format(
                    test_loss_original, test_accuracy_original, test_jaccard, test_hamming, test_score_original_raw
                )
            )
        else:
            print(
                "\n\t [Test set]: Loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, AuPRc raw {:.5f}, AuPRc const {:.5f}, Right Reason Loss on Confounded {:.5f}".format(
                    test_loss_original,
                    test_total_right_answer_loss,
                    test_total_right_reason_loss,
                    test_accuracy_original,
                    test_score_original_raw,
                    test_score_original_const,
                    test_right_reason_loss_confounded,
                )
            )

        # test set only confounder
        print("Test only:")

        if use_probabilistic_circuits:
            (
                test_conf_loss,
                test_conf_accuracy,
                test_conf_jaccard,
                test_conf_hamming,
                test_conf_score_wo_conf_in_train_data_raw,
            ) = test_circuit(
                net=net,
                gate=gate,
                cmpe=cmpe,
                debug_mode=True,
                test_loader=iter(for_test_loader_test_only_confounder_wo_conf_in_train_data),
                title="Validation",
                test=dataloaders["train"],
                device=device,
                montecarlo=montecarlo,
            )

            print(
                "\n\t [Test on confounded train data without confounders]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Jaccard Score {:.3f}, Hamming Loss {:.3f}".format(
                    test_conf_loss, test_conf_accuracy, test_conf_score_wo_conf_in_train_data_raw, test_conf_jaccard, test_conf_hamming
                )
            )

        else:
            test_conf_loss, test_conf_accuracy, test_conf_score_wo_conf_in_train_data_raw, test_conf_score_wo_conf_in_train_data_const, _, _ = test_step(
                net=net,
                test_loader=iter(for_test_loader_test_only_confounder_wo_conf_in_train_data),
                cost_function=cost_function,
                title="Test",
                test=dataloaders["test"],
                device=device,
                debug_mode=True,
                prediction_treshold=prediction_treshold,
                force_prediction=force_prediction,
                superclasses_number=dataloaders["train_set"].n_superclasses,
                montecarlo=montecarlo,
            )

            print(
                "\n\t [Test on confounded train data without confounders]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Area under Precision-Recall Curve const {:.3f}".format(
                    test_conf_loss, test_conf_accuracy, test_conf_score_wo_conf_in_train_data_raw, test_conf_score_wo_conf_in_train_data_const
                )
            )

        if use_probabilistic_circuits:
            (
                test_conf_loss,
                test_conf_accuracy,
                test_conf_jaccard,
                test_conf_hamming,
                test_conf_score_wo_conf_test_data_raw,
            ) = test_circuit(
                net=net,
                gate=gate,
                cmpe=cmpe,
                test_loader=iter(for_test_loader_test_only_confounder_wo_conf),
                title="Test",
                test=dataloaders["test"],
                device=device,
                debug_mode=True,
                montecarlo=montecarlo,
            )

            print(
                "\n\t [Test set Confounder Only WO confounders]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Jaccard Score {:.3f}, Hamming Loss {:.3f}".format(
                    test_conf_loss, test_conf_accuracy, test_conf_score_wo_conf_test_data_raw, test_conf_jaccard, test_conf_hamming
                )
            )

        else:
            test_conf_loss, test_conf_accuracy, test_conf_score_wo_conf_test_data_raw, test_conf_score_wo_conf_test_data_const, _, _ = test_step(
                net=net,
                test_loader=iter(for_test_loader_test_only_confounder_wo_conf),
                cost_function=cost_function,
                title="Test",
                test=dataloaders["test"],
                device=device,
                debug_mode=True,
                prediction_treshold=prediction_treshold,
                force_prediction=force_prediction,
                superclasses_number=dataloaders["train_set"].n_superclasses,
                montecarlo=montecarlo,
            )

            print(
                "\n\t [Test set Confounder Only WO confounders]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Area under Precision-Recall Curve const {:.3f}".format(
                    test_conf_loss, test_conf_accuracy, test_conf_score_wo_conf_test_data_raw, test_conf_score_wo_conf_test_data_const
                )
            )

        if use_probabilistic_circuits:
            (
                test_conf_loss,
                test_conf_accuracy,
                test_conf_jaccard,
                test_conf_hamming,
                test_conf_score_only_conf_raw,
            ) = test_circuit(
                net=net,
                gate=gate,
                cmpe=cmpe,
                test_loader=iter(test_only_confounder),
                title="Test",
                test=dataloaders["test"],
                device=device,
                debug_mode=True,
                montecarlo=montecarlo,
            )

            print(
                "\n\t [Test set Confounder Only]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Jaccard Score {:.3f}, Hamming Loss {:.3f}".format(
                    test_conf_loss, test_conf_accuracy, test_conf_score_only_conf_raw, test_conf_jaccard, test_conf_hamming
                )
            )

        else:
            test_conf_loss, test_conf_accuracy, test_conf_score_only_conf_raw, test_conf_score_only_conf_const, _, _ = test_step(
                net=net,
                test_loader=iter(test_only_confounder),
                cost_function=cost_function,
                title="Test",
                test=dataloaders["test"],
                device=device,
                debug_mode=True,
                prediction_treshold=prediction_treshold,
                force_prediction=force_prediction,
                superclasses_number=dataloaders["train_set"].n_superclasses,
                montecarlo=montecarlo,
            )

            print(
                "\n\t [Test set Confounder Only]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Area under Precision-Recall Curve const {:.3f}".format(
                    test_conf_loss, test_conf_accuracy, test_conf_score_only_conf_raw, test_conf_score_only_conf_const
                )
            )

        # test set
        test_lab_conf_score_raw = 0
        if len(label_confounders[dataset]) > 0:
            if use_probabilistic_circuits:
                (
                    test_lab_conf_loss,
                    test_lab_conf_accuracy,
                    test_lab_conf_jaccard,
                    test_lab_conf_hamming,
                    test_lab_conf_score_raw,
                ) = test_circuit(
                    net=net,
                    gate=gate,
                    cmpe=cmpe,
                    test_loader=iter(labels_only_test_loader),
                    title="Test",
                    test=dataloaders["test"],
                    device=device,
                    montecarlo=montecarlo,
                )

                print(
                    "\n\t [Test set Label confounder Only]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Jaccard Score {:.3f}, Hamming Loss {:.3f}".format(
                        test_lab_conf_loss, test_lab_conf_accuracy, test_lab_conf_score_raw, test_lab_conf_jaccard, test_lab_conf_hamming
                    )
                )

            else:
                test_lab_conf_loss, test_lab_conf_accuracy, test_lab_conf_score_raw, test_lab_score_const, _, _ = test_step(
                    net=net,
                    test_loader=iter(labels_only_test_loader),
                    cost_function=cost_function,
                    title="Test",
                    test=dataloaders["test"],
                    device=device,
                    prediction_treshold=prediction_treshold,
                    force_prediction=force_prediction,
                    superclasses_number=dataloaders["train_set"].n_superclasses,
                    montecarlo=montecarlo,
                )

                print(
                    "\n\t [Test set Label confounder Only]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve raw {:.3f}, Area under Precision-Recall Curve const {:.3f}".format(
                        test_lab_conf_loss, test_lab_conf_accuracy, test_lab_conf_score_raw, test_lab_score_const
                    )
                )

        # save the model if the results are better
        current_score = test_jaccard if use_probabilistic_circuits else test_score_original_const
        if best_test_score > current_score:
            # save the best score
            best_test_score = current_score
            # save the model state of the debugged network
            torch.save(
                net.state_dict(),
                os.path.join(
                    model_folder,
                    "debug_{}_{}.pth".format(
                        network,
                        "integrated_gradients"
                        if integrated_gradients
                        else "input_gradients",
                    ),
                ),
            )
            if use_probabilistic_circuits:
                torch.save(
                    gate.state_dict(),
                    os.path.join(
                        model_folder,
                        "debug_gate_{}_{}.pth".format(
                            network,
                            "integrated_gradients"
                            if integrated_gradients
                            else "input_gradients",
                        ),
                    ),
                )

        print("Done with debug for iteration number: {}".format(iterations))

        print("-----------------------------------------------------")

        if use_probabilistic_circuits:
            logs = {
                "train/train_loss": train_total_loss,
                "train/train_right_anwer_loss": train_total_right_answer_loss,
                "train/train_right_reason_loss": train_total_right_reason_loss,
                "train/train_accuracy": train_total_accuracy,
                "train/train_auprc_raw": train_total_score_raw,
                "train/train_hamming_loss": train_hamming,
                "train/train_jaccard": train_jaccard,
                "train/train_confounded_samples_only_right_reason": train_right_reason_loss_confounded,
                "val/val_loss": val_loss,
                "val/val_accuracy": val_accuracy,
                "val/val_auprc_raw": val_score_raw,
                "val/val_hamming_loss": val_hamming,
                "val/val_jaccard": val_jaccard,
                "test/only_training_confounded_classes_without_confounders_auprc_raw": test_conf_score_wo_conf_in_train_data_raw,
                "test/only_test_confounded_casses_without_confounders_auprc_raw": test_conf_score_wo_conf_test_data_raw,
                "test/only_test_confounded_classes_auprc_raw": test_conf_score_only_conf_raw,
                "test/test_loss": test_loss_original,
                "test/test_right_answer_loss": test_total_right_answer_loss,
                "test/test_right_reason_loss": test_total_right_reason_loss,
                "test/test_accuracy": test_accuracy_original,
                "test/test_auprc_raw": test_score_original_raw,
                "test/test_hamming_loss": test_hamming,
                "test/test_jaccard": test_jaccard,
                "learning_rate": get_lr(optimizer),
            }
        else:
            logs = {
                "train/train_loss": train_total_loss,
                "train/train_right_anwer_loss": train_total_right_answer_loss,
                "train/train_right_reason_loss": train_total_right_reason_loss,
                "train/train_accuracy": train_total_accuracy,
                "train/train_auprc_raw": train_total_score_raw,
                "train/train_auprc_const": train_total_score_const,
                "train/train_confounded_samples_only_right_reason": train_right_reason_loss_confounded,
                "val/val_loss": val_loss,
                "val/val_accuracy": val_accuracy,
                "val/val_auprc_raw": val_score_raw,
                "val/val_auprc_const": val_score_const,
                "test/only_training_confounded_classes_without_confounders_auprc_raw": test_conf_score_wo_conf_in_train_data_raw,
                "test/only_test_confounded_casses_without_confounders_auprc_raw": test_conf_score_wo_conf_test_data_raw,
                "test/only_test_confounded_classes_auprc_raw": test_conf_score_only_conf_raw,
                "test/test_loss": test_loss_original,
                "test/test_right_answer_loss": test_total_right_answer_loss,
                "test/test_right_reason_loss": test_total_right_reason_loss,
                "test/test_accuracy": test_accuracy_original,
                "test/test_auprc_raw": test_score_original_raw,
                "test/test_auprc_const": test_score_original_const,
                "learning_rate": get_lr(optimizer),
            }

            if train_loss_parent is not None and train_loss_children is not None:
                logs.update({"train/train_right_answer_loss_parent": train_loss_parent})
                logs.update({"train/train_right_answer_loss_children": train_loss_children})

            if val_loss_parent is not None and val_loss_children is not None:
                logs.update({"val/val_right_answer_loss_parent": val_loss_parent})
                logs.update({"val/val_right_answer_loss_children": val_loss_children})

            if test_loss_parent is not None and test_loss_children is not None:
                logs.update({"test/test_right_answer_loss_parent": test_loss_parent})
                logs.update({"test/test_right_answer_loss_children": test_loss_children})

            if len(label_confounders[dataset]) > 0:
                logs.update({"test/only_label_confounders_auprc_raw": test_lab_conf_score_raw})

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                logs
            )

        # log on tensorboard
        for key, value in logs.items():
            log_value(writer, it + 1, value, key)

        # scheduler step
        scheduler.step(val_loss)


    # save some test confounded examples
    save_some_confounded_samples(
        dataset=dataset,
        net=net,
        start_from=0,
        number=30,
        dataloaders=dataloaders,
        device=device,
        folder=debug_folder,
        integrated_gradients=integrated_gradients,
        loader=print_iterator_after,
        prefix="after",
        prediction_treshold=prediction_treshold,
        force_prediction=force_prediction,
        use_softmax=use_softmax,
        use_circuits=use_probabilistic_circuits,
        gate=gate,
        cmpe=cmpe
    )

    # give the correlation
    #  compute_gradient_confound_correlation(
    #      net,
    #      test_debug,
    #      integrated_gradients,
    #      100,
    #      debug_folder,
    #      "end_debug_test",
    #      device,
    #  )


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the debug of the network
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("debug", help="Debug network subparser")
    parser.add_argument("exp_name", type=str, help="name of the experiment")
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["resnet", "lenet",  "lenet7", "alexnet", "mlp"],
        default="resnet",
        help="Network",
    )
    parser.add_argument(
        "--weights-path-folder",
        "-wpf",
        type=str,
        default="models",
        help="Path where to load the best.pth file",
    )
    parser.add_argument("--iterations", "-it", type=int, default=30, help="Debug Epocs")
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="Train batch size"
    )
    parser.add_argument(
        "--debug-folder", "-df", type=str, default="debug", help="Debug folder"
    )
    parser.add_argument(
        "--model-folder",
        "-mf",
        type=str,
        default="models",
        help="Folder where to save the model",
    )
    parser.add_argument(
        "--test-batch-size", "-tbs", type=int, default=128, help="Test batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--rrr-regularization-rate",
        type=float,
        default=100.0,
        help="RRR regularization rate",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument(
        "--integrated-gradients",
        "-igrad",
        dest="integrated_gradients",
        action="store_true",
        help="Use integrated gradients [default]",
    )
    parser.add_argument(
        "--no-integrated-gradients",
        "-noigrad",
        dest="integrated_gradients",
        action="store_false",
        help="Use input gradients",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device",
    )
    parser.add_argument(
        "--project", "-w", type=str, default="chmcnn-project", help="wandb project"
    )
    parser.add_argument(
        "--entity", "-e", type=str, default="samu32", help="wandb entity"
    )
    parser.add_argument(
        "--gradient-analysis",
        "-grad-show",
        dest="gradient_analysis",
        action="store_true",
        help="See the gradient behaviour",
    )
    parser.add_argument(
        "--no-gradient-analysis",
        "-no-grad-show",
        dest="gradient_analysis",
        action="store_false",
        help="Do not see the gradient behaviour",
    )
    parser.add_argument("--wandb", "-wdb", type=bool, default=False, help="wandb")
    parser.add_argument(
        "--constrained-layer",
        "-clayer",
        dest="constrained_layer",
        action="store_true",
        help="Use the Giunchiglia et al. layer to enforce hierarchical logical constraints",
    )
    parser.add_argument(
        "--no-constrained-layer",
        "-noclayer",
        dest="constrained_layer",
        action="store_false",
        help="Do not use the Giunchiglia et al. layer to enforce hierarchical logical constraints",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="dataloaders num workers"
    )
    parser.add_argument(
        "--prediction-treshold", type=float, default=0.5, help="considers the class to be predicted in a multilabel classification setting"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="scheduler patience"
    )
    parser.add_argument(
        "--force-prediction",
        "-fpred",
        dest="force_prediction",
        action="store_true",
        help="Force the prediction",
    )
    parser.add_argument(
        "--no-force-prediction",
        "-nofspred",
        dest="force_prediction",
        action="store_false",
        help="Use the classic prediction output logits",
    )
    parser.add_argument(
        "--fixed-confounder",
        "-fixconf",
        dest="fixed_confounder",
        action="store_true",
        help="Force the confounder position to the bottom right",
    )
    parser.add_argument(
        "--no-fixed-confounder",
        "-nofixconf",
        dest="fixed_confounder",
        action="store_false",
        help="Let the confounder be placed in a random position in the image",
    )
    parser.add_argument(
        "--use-softmax",
        "-soft",
        dest="use_softmax",
        action="store_true",
        help="Force the confounder position to use softmax as loss",
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar", choices=["mnist", "cifar", "fashion", "omniglot"], help="which dataset to use"
    )
    parser.add_argument(
        "--simplified-dataset",
        "-simdat",
        dest="simplified_dataset",
        action="store_true",
        help="If possibile, use a simplified version of the dataset",
    )
    parser.add_argument(
        "--imbalance-dataset",
        "-imdat",
        dest="imbalance_dataset",
        action="store_true",
        help="Imbalance the dataset introducing another type of confounding",
    )
    parser.add_argument('-balsub','--balance-subclasses', nargs='+', help='Which subclasses to balance', type=str)
    parser.add_argument('-balweight','--balance-weights', nargs='+', help='Float weights used so as to balance the subclasses', type=float)
    parser.add_argument(
        "--correct-by-duplicating-samples",
        "-cbds",
        dest="correct_by_duplicating_samples",
        action="store_true",
        help="Correct the model by duplicating samples instead of upweighting"
    )
    parser.add_argument(
        "--use-probabilistic-circuits",
        "-probcirc",
        action="store_true",
        help="Whether to use the probabilistic circuit instead of the Giunchiglia approach",
    )
    parser.add_argument(
        "--gates",
        type=int,
        default=1,
        help="Number of hidden layers in gating function (default: 1)"
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=1,
        help="Number of hidden layers in gating function (default: 1)"
    )
    parser.add_argument(
        "--S",
        type=int,
        default=0,
        help="PSDD scaling factor (default: 0)"
    )
    parser.add_argument(
        "--constraint-folder",
        type=str,
        default="./constraints",
        help="Folder for storing the constraints"
    )
    parser.add_argument(
        "--use-gate-output",
        "-ugo",
        action="store_true",
        help="Whether to use the gate output for the RRR loss"
    )
    parser.add_argument(
        "--montecarlo",
        "-mnt",
        action="store_true",
        help="Use a montecarlo approach for the predictions"
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(
        func=main,
        integrated_gradients=True,
        gradient_analysis=False,
        constrained_layer=True,
        force_prediction=False,
        fixed_confounder=False,
        use_softmax=False,
        simplified_dataset=False,
        imbalance_dataset=False,
        balance_subclasses=[],
        balance_weights=[],
        correct_by_duplicating_samples=False,
        use_gate_output=False,
        montecarlo=False
    )


def check_balance_dataset_integrity(
    dataset: str,
    subclasses_list: List[str],
    weights_list: List[float]
) -> None:
    """Checks whether the balance arguments of the dataset are compliant with the integrity constraint or not
    Args:
        dataset [str]: datset name
        subclasses_list [str]: list of subclasses to rebalance
        weights_list [str]: list of weigths used to rebalance
    Asserts:
        error if the dataset hierarchy is not respected
        error if weights for certain classes are not specified
    """
    # Skip if the arrays are empty
    if len(subclasses_list) == 0 and len(weights_list) == 0:
        return
    assert len(subclasses_list) == len(weights_list), "Subclass and weights lists must have equal length"
    _, children, _ = get_confounders_and_hierarchy(dataset)
    children = list(itertools.chain.from_iterable(children))
    for cl in subclasses_list:
        assert cl in children, f"{cl} is not in the children classes which are recognized by the hierarchy"
    for i, w in enumerate(weights_list):
        assert w > 0, f"{w}, namely the weight associated to {subclasses_list[i]} cannot be negative"


def main(args: Namespace) -> None:
    """Checks the command line arguments and then runs the debug

    Args:
        args (Namespace): command line arguments
    """
    print("\n### Network debug ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # check balance integrity
    check_balance_dataset_integrity(args.dataset, args.balance_subclasses, args.balance_weights)

    # set the seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # img_size
    img_size = 32
    img_depth = 3
    output_classes = 121

    if args.dataset == "mnist":
        img_size = 28
        img_depth = 1
        output_classes = 67
    elif args.dataset == "fashion":
        img_size = 28
        img_depth = 1
        output_classes = 10
    elif args.dataset == "omniglot":
        img_size = 32
        img_depth = 1
        output_classes = 680

    if args.network == "alexnet":
        img_size = 224

    # Load dataloaders
    dataloaders = load_dataloaders(
        dataset_type=args.dataset,
        img_size=img_size,  # the size is squared
        img_depth=img_depth,  # number of channels
        device=args.device,
        csv_path="./dataset/train.csv",
        test_csv_path="./dataset/test_reduced.csv",
        val_csv_path="./dataset/val.csv",
        cifar_metadata="./dataset/pickle_files/meta",
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        normalize=True,  # normalize the dataset
        num_workers=args.num_workers,
        fixed_confounder=args.fixed_confounder,
        simplified_dataset=args.simplified_dataset,
        imbalance_dataset=args.imbalance_dataset
    )

    # Load dataloaders
    print("Load network weights...")

    # Network
    if args.network == "lenet":
        net = LeNet5(
            dataloaders["train_R"], output_classes, args.constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    elif args.network == "lenet7":
        net = LeNet7(
            dataloaders["train_R"], output_classes, args.constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    elif args.network == "alexnet":
        net = AlexNet(
            dataloaders["train_R"], output_classes, args.constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    elif args.network == "mlp":
        net = MLP(
            dataloaders["train_R"], output_classes, args.constrained_layer, channels=img_depth, img_height=img_size, img_width=img_size
        )  # 20 superclasses, 100 subclasses + the root
    else:
        net = ResNet18(
            dataloaders["train_R"], 121, args.constrained_layer
        )  # 20 superclasses, 100 subclasses + the root

    # move everything on the cpu
    net = net.to(args.device)
    net.R = net.R.to(args.device)
    # zero grad
    net.zero_grad()
    # summary
    summary(net, (img_depth, img_size, img_size))

    # set up tensorboard
    log_directory = "runs/exp_{}".format(args.exp_name)

    # create a logger for the experiment
    print("Creating writer in {}".format(log_directory))
    writer = SummaryWriter(log_dir=log_directory)

    # set wandb if needed
    if args.wandb:
        # Log in to your W&B account
        wandb.login()
        # set the argument to true
        args.set_wandb = args.wandb
    else:
        args.set_wandb = False

    if args.wandb:
        # start the log
        wandb.init(project=args.project, entity=args.entity)

    # use the probabilistic circuit
    cmpe: CircuitMPE = None
    gate: DenseGatingFunction = None

    if args.use_probabilistic_circuits:
        print("Using probabilistic circuits...")
        cmpe, gate = prepare_probabilistic_circuit(
            dataloaders['train_set'].get_A(),
            args.constraint_folder,
            args.dataset,
            args.device,
            args.gates,
            args.num_reps,
            output_classes,
            args.S,
        )

    # optimizer
    optimizer = get_adam_optimizer(
        net, args.learning_rate, weight_decay=args.weight_decay
    )
    if args.use_probabilistic_circuits:
        optimizer = get_sdg_optimizer_with_gate(net, gate, args.learning_rate, weight_decay=args.weight_decay)
        #  optimizer = get_adam_optimizer_with_gate(net, gate, args.learning_rate, weight_decay=args.weight_decay)

    # scheduler
    scheduler = get_plateau_scheduler(optimizer=optimizer, patience=args.patience)

    # Test on best weights (of the confounded model)
    load_best_weights(net, args.weights_path_folder, args.device)
    if args.use_probabilistic_circuits:
        load_best_weights_gate(gate, args.weights_path_folder, args.device)

    # dataloaders
    test_loader = dataloaders["test_loader"]
    val_loader = dataloaders["val_loader_debug_mode"]

    # weights for the cross entropy
    binary_cc_revise = BCELoss()

    # if I have to use upweighting then compute the weights
    if len(args.balance_subclasses) != 0 and not args.correct_by_duplicating_samples:
        weights = torch.ones(output_classes - 1)
        weights_idx = get_hierarchical_index_from_named_label(
            output_classes - 1,
            args.balance_subclasses,
            dataloaders["train_set"].nodes_names_without_root
        )
        for i, el in enumerate(args.balance_weights):
            weights[weights_idx[i]] = el
        print("Weights: ", weights)
        # to device
        weights = weights.to(args.device)
        binary_cc_revise = BCELoss(weight=weights)

    # define the cost function (binary cross entropy for the current models)
    cost_function = torch.nn.BCELoss()

    # test set
    if args.use_probabilistic_circuits:
        (
            test_loss,
            test_accuracy,
            test_jaccard,
            test_hamming,
            test_score
        )= test_circuit(
            net=net,
            gate=gate,
            cmpe=cmpe,
            test_loader=iter(test_loader),
            title="Test",
            test=dataloaders["test"],
            device=args.device,
            montecarlo=args.montecarlo,
        )
    else:
        test_loss, test_accuracy, test_score_raw, test_score_const, _, _ = test_step(
            net=net,
            test_loader=iter(test_loader),
            cost_function=cost_function,
            title="Test",
            test=dataloaders["test"],
            device=args.device,
            prediction_treshold=args.prediction_treshold,
            force_prediction=args.force_prediction,
            superclasses_number=dataloaders["train_set"].n_superclasses,
            montecarlo=args.montecarlo,
        )

    # load the human readable labels dataloader
    test_loader_with_label_names = dataloaders["test_loader_with_labels_name"]
    labels_name = dataloaders["test_set"].nodes_names_without_root
    # load the training dataloader
    training_loader_with_labels_names = dataloaders["training_loader_with_labels_names"]

    # collect stats
    if args.use_probabilistic_circuits:
        (
            _,
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
            clf_report,  # classification matrix
            y_test,      # ground-truth for multiclass classification matrix
            y_pred,      # predited values for multiclass classification matrix
            _,
        ) = test_step_with_prediction_statistics_with_gates(
            net=net,
            gate=gate,
            cmpe=cmpe,
            nodes=dataloaders["test_set"].get_nodes(),
            test_loader=iter(test_loader_with_label_names),
            title="Collect Statistics",
            test=dataloaders["test"],
            device=args.device,
            labels_name=labels_name,
            montecarlo=args.montecarlo,
        )
    else:
        (
            _,
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
            clf_report,  # classification matrix
            y_test,      # ground-truth for multiclass classification matrix
            y_pred,      # predited values for multiclass classification matrix
            _,
            _
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(test_loader_with_label_names),
            nodes=dataloaders["test_set"].get_nodes(),
            cost_function=cost_function,
            title="Collect Statistics",
            test=dataloaders["test"],
            device=args.device,
            labels_name=labels_name,
            prediction_treshold=args.prediction_treshold,
            force_prediction=args.force_prediction,
            superclasses_number=dataloaders["train_set"].n_superclasses,
            montecarlo=args.montecarlo,
        )

    # confusion matrix before debug
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 30),
        fig_name="{}/before_confusion_matrix".format(args.debug_folder),
        normalize=False,
    )
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 30),
        fig_name="{}/before_confusion_matrix_normalized".format(args.debug_folder),
        normalize=True,
    )
    plot_confusion_matrix_statistics(
        clf_report=clf_report,
        fig_name="{}/before_confusion_matrix_statistics.pdf".format(
            args.debug_folder
        ),
    )
    prediction_statistics_boxplot(
        statistics_predicted,
        statistics_correct,
        args.debug_folder,
        "Before_Overpredicted",
        "Overpredicted or not"
    )

    print("Network resumed, performances:")

    if args.use_probabilistic_circuits:
        print(
            "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Jaccard Score {:.3f}, Test Hamming Loss {:.3f}, Test Area under Precision-Recall Curve Raw {:.3f}".format(
                test_loss, test_accuracy, test_jaccard, test_hamming, test_score
            )
        )
    else:
        print(
            "\n\t [TEST SET]: Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve raw {:.3f}, Test Area under Precision-Recall Curve const {:.3f}".format(
                test_loss, test_accuracy, test_score_raw, test_score_const
            )
        )

    print("-----------------------------------------------------")

    # log values in the first position, dopo mettere apposto con il debug
    logs = {
        "test/test_loss": test_loss,
        "test/test_accuracy": test_accuracy,
    }

    if args.use_probabilistic_circuits:
        logs.update({"test/test_jaccard": test_jaccard})
        logs.update({"test/test_hamming_loss": test_hamming})
        logs.update({"test/test_auprc_raw": test_score})
    else:
        logs.update({"test/test_auprc_raw": test_score_raw})
        logs.update({"test/test_auprc_const": test_score_const})

    for key, value in logs.items():
        log_value(writer, 0, value, key)

    print("#> Debug...")

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.watch(net)

    # choose carefully the kind of loss to employ
    if args.use_probabilistic_circuits:
        reviseLoss = RRRLossWithGate(
            net=net,
            gate=gate,
            cmpe=cmpe,
            regularizer_rate=args.rrr_regularization_rate,
        )
    elif not args.integrated_gradients:
        # rrr loss based on input gradients
        reviseLoss = RRRLoss(
            net=net,
            regularizer_rate=args.rrr_regularization_rate,
            base_criterion=binary_cc_revise,
        )
    else:
        # integrated gradients RRRLoss
        reviseLoss = IGRRRLoss(
            net=net,
            regularizer_rate=args.rrr_regularization_rate,
            base_criterion=binary_cc_revise,
        )

    # launch the debug a given number of iterations
    debug(
        net=net, # network
        dataloaders=dataloaders, # dataloader
        cost_function=torch.nn.BCELoss(), # Binary Cross Entropy loss
        optimizer=optimizer, # learning rate optimizer
        scheduler=scheduler, # learning rate scheduler
        title="debug", # title of the iterator
        debug_test_loader=val_loader, # validation loader on which to validate the model
        reviseLoss=reviseLoss, # RRR
        superclasses_number=dataloaders["train_set"].n_superclasses,
        gate=gate,
        cmpe=cmpe,
        writer=writer,
        **vars(args) # additional variables
    )

    print("After debugging...")

    # re-test set
    if args.use_probabilistic_circuits:
        (
            test_loss,
            test_accuracy,
            test_jaccard,
            test_hamming,
            test_score
        )= test_circuit(
            net=net,
            gate=gate,
            cmpe=cmpe,
            test_loader=iter(test_loader),
            title="Test",
            test=dataloaders["test"],
            device=args.device,
            montecarlo=args.montecarlo,
        )
    else:
        test_loss, test_accuracy, test_score_raw, test_score_const, _, _ = test_step(
            net=net,
            test_loader=iter(test_loader),
            cost_function=cost_function,
            title="Test",
            test=dataloaders["test"],
            device=args.device,
            prediction_treshold=args.prediction_treshold,
            force_prediction=args.force_prediction,
            superclasses_number=dataloaders["train_set"].n_superclasses,
            montecarlo=args.montecarlo,
        )

    if args.use_probabilistic_circuits:
        print(
            "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Jaccard Score {:.3f}, Test Hamming Loss {:.3f}, Test Area under Precision-Recall Curve Raw {:.3f}".format(
                test_loss, test_accuracy, test_jaccard, test_hamming, test_score
            )
        )

    else:
        print(
            "\n\t [TEST SET]: Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve raw {:.3f}, Test Area under Precision-Recall Curve const {:.3f}".format(
                test_loss, test_accuracy, test_score_raw, test_score_const
            )
        )

    ## TRAIN ##
    # collect stats
    if args.use_probabilistic_circuits:
        (
            _,
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
            clf_report,  # classification matrix
            y_test,      # ground-truth for multiclass classification matrix
            y_pred,      # predited values for multiclass classification matrix
            _,
        ) = test_step_with_prediction_statistics_with_gates(
            net=net,
            gate=gate,
            cmpe=cmpe,
            nodes=dataloaders["test_set"].get_nodes(),
            test_loader=iter(training_loader_with_labels_names),
            title="Collect Statistics [TRAIN]",
            test=dataloaders["train"],
            device=args.device,
            labels_name=labels_name,
            montecarlo=args.montecarlo,
        )
    else:
        (
            _,
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
            clf_report,  # classification matrix
            y_test,  # ground-truth for multiclass classification matrix
            y_pred,  # predited values for multiclass classification matrix
            _,
            _
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(training_loader_with_labels_names),
            nodes=dataloaders["test_set"].get_nodes(),
            cost_function=cost_function,
            title="Collect Statistics [TRAIN]",
            test=dataloaders["train"],
            device=args.device,
            labels_name=labels_name,
            prediction_treshold=args.prediction_treshold,
            force_prediction=args.force_prediction,
            superclasses_number=dataloaders["train_set"].n_superclasses,
            montecarlo=args.montecarlo,
        )

    ## ! Confusion matrix !
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 30),
        fig_name="{}/train_confusion_matrix_normalized".format(
            os.environ["IMAGE_FOLDER"]
        ),
        normalize=True,
    )
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 30),
        fig_name="{}/train_confusion_matrix".format(os.environ["IMAGE_FOLDER"]),
        normalize=False,
    )
    plot_confusion_matrix_statistics(
        clf_report=clf_report,
        fig_name="{}/train_confusion_matrix_statistics.pdf".format(
            os.environ["IMAGE_FOLDER"]
        ),
    )
    # grouped boxplot
    grouped_boxplot(
        statistics_predicted,
        os.environ["IMAGE_FOLDER"],
        "Predicted",
        "Not predicted",
        "train_predicted",
    )
    grouped_boxplot(
        statistics_correct,
        os.environ["IMAGE_FOLDER"],
        "Correct prediction",
        "Wrong prediction",
        "train_accuracy",
    )

    # collect stats
    if args.use_probabilistic_circuits:
        (
            _,
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
            clf_report,  # classification matrix
            y_test,      # ground-truth for multiclass classification matrix
            y_pred,      # predited values for multiclass classification matrix
            _,
        ) = test_step_with_prediction_statistics_with_gates(
            net=net,
            gate=gate,
            cmpe=cmpe,
            test_loader=iter(test_loader_with_label_names),
            nodes=dataloaders["test_set"].get_nodes(),
            title="Collect Statistics [TEST]",
            test=dataloaders["test"],
            device=args.device,
            labels_name=labels_name,
            montecarlo=args.montecarlo,
        )
    else:
        (
            _,
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
            clf_report,  # classification matrix
            y_test,  # ground-truth for multiclass classification matrix
            y_pred,  # predited values for multiclass classification matrix
            _,
            _
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(test_loader_with_label_names),
            nodes=dataloaders["test_set"].get_nodes(),
            cost_function=cost_function,
            title="Collect Statistics [TEST]",
            test=dataloaders["test"],
            device=args.device,
            labels_name=labels_name,
            prediction_treshold=args.prediction_treshold,
            force_prediction=args.force_prediction,
            superclasses_number=dataloaders["train_set"].n_superclasses,
            montecarlo=args.montecarlo,
        )

    ## confusion matrix after debug
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 20),
        fig_name="{}/test_after_confusion_matrix_normalized".format(
            args.debug_folder
        ),
        normalize=True,
    )
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 20),
        fig_name="{}/test_after_confusion_matrix".format(args.debug_folder),
        normalize=False,
    )
    plot_confusion_matrix_statistics(
        clf_report=clf_report,
        fig_name="{}/test_after_confusion_matrix_statistics.pdf".format(
            args.debug_folder
        ),
    )

    # grouped boxplot (for predictions and accuracy capabilities)
    grouped_boxplot(
        statistics_predicted,
        args.debug_folder,
        "Predicted",
        "Not predicted",
        "test_predicted",
    )
    grouped_boxplot(
        statistics_correct,
        args.debug_folder,
        "Correct prediction",
        "Wrong prediction",
        "test_accuracy",
    )
    prediction_statistics_boxplot(
        statistics_predicted,
        statistics_correct,
        args.debug_folder,
        "After_Overpredicted",
        "Overpredicted or not"
    )

    if args.use_probabilistic_circuits:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            y_test,      # ground-truth for multiclass classification matrix
            y_pred,      # predited values for multiclass classification matrix
            _,
        ) = test_step_with_prediction_statistics_with_gates(
            net=net,
            gate=gate,
            cmpe=cmpe,
            test_loader=iter(dataloaders["test_loader_only_label_confounders_with_labels_names"]),
            nodes=dataloaders["test_set"].get_nodes(),
            title="Computing statistics in label confoundings",
            test=dataloaders["train"],
            device=args.device,
            labels_name=labels_name,
            montecarlo=args.montecarlo,
        )
    else:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            y_test,  # ground-truth for multiclass classification matrix
            y_pred,  # predited values for multiclass classification matrix
            _,
            _,
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(dataloaders["test_loader_only_label_confounders_with_labels_names"]),
            cost_function=cost_function,
            title="Computing statistics in label confoundings",
            nodes=dataloaders["test_set"].get_nodes(),
            test=dataloaders["train"],
            device=args.device,
            labels_name=labels_name,
            prediction_treshold=args.prediction_treshold,
            force_prediction=args.force_prediction,
            use_softmax=args.use_softmax,
            superclasses_number=dataloaders["train_set"].n_superclasses,
            montecarlo=args.montecarlo,
        )

    labels_predictions_dict, counter_dict = prepare_dict_label_predictions_from_raw_predictions(y_pred, y_test, labels_name, args.dataset, True)
    plot_confounded_labels_predictions(
        labels_predictions_dict,
        counter_dict,
        args.debug_folder,
        "imbalancing_predictions",
        args.dataset
    )

    # save the model state of the debugged network
    torch.save(
        net.state_dict(),
        os.path.join(
            args.model_folder,
            "after_training_debug_{}_{}.pth".format(
                args.network,
                "integrated_gradients"
                if args.integrated_gradients
                else "input_gradients",
            ),
        ),
    )
    if args.use_probabilistic_circuits:
        torch.save(
            gate.state_dict(),
            os.path.join(
                args.model_folder,
                "after_training_debug_gate_{}_{}.pth".format(
                    args.network,
                    "integrated_gradients"
                    if args.integrated_gradients
                    else "input_gradients",
                ),
            ),
    )

    # close wandb
    if args.wandb:
        # finish the log
        wandb.finish()
    # close writer
    writer.close()
