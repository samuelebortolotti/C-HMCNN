"""Utils module"""
import torch
import torch.nn as nn
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from typing import Tuple, Dict, Optional, List, Any
from torch.utils import tensorboard
import torch.nn.functional as F
import seaborn as sns
from chmncc.config import (
    mnist_hierarchy,
    cifar_hierarchy,
    omniglot_hierarchy,
    omniglot_confunders,
    mnist_confunders,
    cifar_confunders,
    fashion_hierarchy,
    fashion_confunders,
    label_confounders,
)
from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE

#  from pysdd.sdd import SddManager, Vtree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pypsdd"))

from pypsdd import io

#  from pypsdd import SddManager, Vtree
from pysdd.sdd import SddManager, Vtree

import networkx as nx


################### Dotdict ##################


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


################### Get Lr  ##################


def get_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    r"""
    Function which returns the learning rate value
    used in the optimizer

    Args:
        optimizer [torch.optim.Optimizer]: optimizer

    Returns:
        lr [float]: learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


################### Logging ###################


def log_images(
    writer: tensorboard.SummaryWriter,
    img: torch.Tensor,
    epoch: int,
    title: str,
) -> None:
    r"""
    Log images on the summary writer, this function is usefull during
    debug sessions.

    Args:
        writer [tensorboard.SummaryWriter]: summary writer
        img [torch.Tensor]: image to log
        title [str]: title of the log
    """
    try:
        # Log training images in a row of 8
        logged_img = torchvision.utils.make_grid(
            img.data, nrow=8, normalize=True, scale_each=True
        )
        # add image
        writer.add_image(title, logged_img, epoch)
    except Exception as e:
        print("Couldn't log results: {}".format(e))


def log_values(
    writer: tensorboard.SummaryWriter,
    step: int,
    loss: float,
    accuracy: float,
    prefix: str,
) -> None:
    r"""
    Function which writes the loss and the accuracy of a model by the means of
    a SummaryWriter

    Args:
        writer [tensorboard.SummaryWriter]: summary writer
        step [int]: current step
        loss [float]: current loss
        accuracy [float]: accuracy
        prefix [str]: log prefix
    """
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


def log_value(
    writer: tensorboard.SummaryWriter,
    step: int,
    value: float,
    string: str,
) -> None:
    r"""
    Function which writes the loss and the accuracy of a model by the means of
    a SummaryWriter

    Args:
        writer [tensorboard.SummaryWriter]: summary writer
        step [int]: current step
        value [float]: value
        string [str]: log string
    """
    writer.add_scalar(f"{string}/loss", value, step)


################### Resume the network ###################


def load_best_weights(net: nn.Module, exp_name: str, device: str) -> None:
    r"""
    Function which loads the best weights of the network, basically it
    looks for the `{exp_name}/best.pth` and loads it

    Args:
        net [nn.Module]: network architecture
        exp_name [str]: folder name
        device [str]: device name
    """
    best_file = os.path.join(exp_name, "best.pth")
    print(best_file)
    if os.path.isfile(best_file):
        checkpoint = torch.load(best_file, map_location=torch.device(device))
        print("#> Resume best")
        net.load_state_dict(checkpoint)
    else:
        print("## Not Resumed ##")


def load_best_weights_gate(
    gate: DenseGatingFunction, exp_name: str, device: str
) -> None:
    r"""
    Function which loads the best weights of the gating function, basically it
    looks for the `{exp_name}/best_gate.pth` and loads it

    Args:
        gate [DenseGatingFunction]: gate function
        exp_name [str]: folder name
        device [str]: device name
    """
    best_file = os.path.join(exp_name, "best_gate.pth")
    if os.path.isfile(best_file):
        checkpoint = torch.load(best_file, map_location=torch.device(device))
        print("#> Resume best gate")
        gate.load_state_dict(checkpoint)
    else:
        print("## Gate Not Resumed ##")


def load_last_weights(net: nn.Module, exp_name: str, device: str) -> None:
    r"""
    Function which loads the last weights of the network, basically it
    looks for the `{exp_name}/net.pth` and loads it

    Args:
        net [nn.Module]: network architecture
        exp_name [str]: folder name
        device [str]: device name
    """
    best_file = os.path.join(exp_name, "last.pth")
    if os.path.isfile(best_file):
        checkpoint = torch.load(best_file, map_location=torch.device(device))
        print("#> Resume last")
        net.load_state_dict(checkpoint)
        net.eval()
    else:
        print("## Not Resumed ##")


def load_last_weights_gate(
    gate: DenseGatingFunction, exp_name: str, device: str
) -> None:
    r"""
    Function which loads the last weights of the gating function, basically it
    looks for the `{exp_name}/last_gate.pth` and loads it

    Args:
        gate [DenseGatingFunction]: gate function
        exp_name [str]: folder name
        device [str]: device name
    """
    best_file = os.path.join(exp_name, "last_gate.pth")
    if os.path.isfile(best_file):
        checkpoint = torch.load(best_file, map_location=torch.device(device))
        print("#> Resume last gate")
        gate.load_state_dict(checkpoint)
    else:
        print("## Gate Not Resumed ##")


def resume_training(
    resume_training: bool,
    experiment_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gate: DenseGatingFunction = None,
) -> Tuple[Dict, Dict, int]:
    r"""
    Resumes the training if the corresponding flag is specified.
    If the resume_training flag is set to true, the function tries to recover, from the
    checkpoint specified, the number of epoch, training and validation parameters.
    If the resume_training flag is set to false, the parameters are set to the default ones

    Args:
        model [nn.Module]: network architecture
        experiment_name [str]: where the weight file is located
        optimizer [torch.optim.Optimizer]: optimizer

    Returns:
        training_params [Dict]: training parameters
        val_params [Dict]: validation parameters [step, best_loss]
        start_epoch [int]: start epoch number
    """
    if resume_training:
        print(f"#> Recovering the last paramters at {experiment_name}/ckpt.pth")
        resumef = os.path.join(experiment_name, "ckpt.pth")
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("#> Resuming previous training")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            gate.load_state_dict(checkpoint["gate"])
            training_params = checkpoint["training_params"]
            start_epoch = training_params["start_epoch"] + 1
            val_params = checkpoint["val_params"]
            print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint["optimizer"]["param_groups"]))
            print("==> checkpoint['training_params']")
            for k in checkpoint["training_params"]:
                print("\t{}, {}".format(k, checkpoint["training_params"][k]))
            print("==> checkpoint['val_params']")
            for k in checkpoint["val_params"]:
                print("\t{}, {}".format(k, checkpoint["val_params"][k]))
        else:
            raise Exception(
                "Couldn't resume training with checkpoint {}".format(resumef)
            )
    else:
        start_epoch = 0
        training_params = {}
        val_params = {}
        # worst possible auc score
        val_params["best_score"] = -np.inf

    return training_params, val_params, start_epoch


########## GET CONSTRAINED OUTPUT ######################:


def get_constr_out(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Given the output of the neural network x returns the output of MCM given the hierarchy constraint
    expressed in the matrix R. This is employed only during the evaluation/test phase of the network

    Args:
        x [torch.tensor]: output of the neural network
        R [torch.tensor]: matrix of the ancestors

    Returns:
        final_out [torch.tensor]: output constrained
    """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out


def get_constr_indexes(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Given the output of the neural network x returns the indexes of MCM hierarchy which have influenced
    the final prediction (implication constrained).

    Args:
        x [torch.tensor]: output of the neural network
        R [torch.tensor]: matrix of the ancestors

    Returns:
        final_out [torch.tensor]: output constrained
    """
    c_out = x.double()
    torch.set_printoptions(profile="full")
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    _, final_indexes = torch.max(R_batch * c_out.double(), dim=2)
    return final_indexes


def average_image_contributions(image: np.ndarray) -> np.ndarray:
    """Returns a black and white image making the average contribution of each channel

    Args:
        image [np.ndarray]: numpy array of size [size1, ...., channels]
    Returns:
        average [np.ndarray]: average image
    """
    return np.mean(image, axis=image.ndim - 1)


def average_image_contributions_tensor(image: torch.Tensor) -> torch.Tensor:
    """Returns a black and white image making the average contribution of each channel

    Args:
        image [torch.Tensor]: torch array of size [size1, ...., channels]
    Returns:
        average [torch.Tensor]: average image
    """
    return torch.mean(image, dim=0)


def prediction_statistics_boxplot(
    statistics: Dict[str, List[int]],
    correct: Dict[str, List[int]],
    image_folder: str,
    statistics_name: str,
    title: str,
) -> None:
    """Grouped Boxplot
    print the grouped boxplot for the statistics

    Args:
      statistics [Dict[List[int]]]: set of statistics
      correct: Dict[str, List[int]]: set of correct and incorrect sample
      image_folder [str]: image folder
      statistics_name [str]: name of the statistics
      title [str]: title
    """
    predicted_perc = {}
    # create the statistics
    for key, item in statistics.items():
        if key != "total":
            predicted_perc[key] = float(statistics[key][1]) / (
                correct[key][0] + correct[key][1]
            )

    fig = plt.figure(figsize=(16, 8))
    plt.bar(range(len(predicted_perc)), list(predicted_perc.values()), align="center")
    plt.xticks(range(len(predicted_perc)), list(predicted_perc.keys()))
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.title("{}".format(title))
    fig.savefig("{}/statistics_{}_total.png".format(image_folder, statistics_name))
    plt.close(fig)
    print("{}/statistics_{}_total.png".format(image_folder, statistics_name))
    print("{}".format(predicted_perc))


def grouped_boxplot(
    statistics: Dict[str, List[int]],
    image_folder: str,
    correct_txt: str,
    wrong_txt: str,
    statistics_name: str,
) -> None:
    """Grouped Boxplot
    print the grouped boxplot for the statistics

    Args:
      statistics [Dict[List[int]]]: set of statistics
      image_folder [str]: image folde
    """
    predicted = []
    unpredicted = []
    index = []
    # create the statistics
    for key, item in statistics.items():
        if key != "total":
            predicted.append(item[1])
            unpredicted.append(item[0])
            index.append(key)

    fig = plt.figure(figsize=(8, 4))
    titles = np.array([correct_txt, wrong_txt])
    values = np.array([statistics["total"][1], statistics["total"][0]])
    plot = pd.Series(values).plot(kind="bar", color=["green", "red"])
    plot.bar_label(plot.containers[0], label_type="edge")
    plot.set_xticklabels(titles)
    plt.xticks(rotation=0)
    plt.title("Total: {} vs {}".format(correct_txt, wrong_txt))
    plt.tight_layout()
    fig.savefig("{}/statistics_{}_total.png".format(image_folder, statistics_name))
    plt.close(fig)

    # print data
    #  for i, (el_p, el_u, el_i) in enumerate(
    #      zip(split(predicted, 10), split(unpredicted, 10), split(index, 10))
    #  ):
    #      # data
    #      data = {correct_txt: el_p, wrong_txt: el_u}
    #      # figure
    #      df = pd.DataFrame(data, index=el_i)
    #      plot = df.plot.bar(rot=0, figsize=(11, 9), color=["green", "red"])
    #      plot.bar_label(plot.containers[0], label_type="edge")
    #      plot.bar_label(plot.containers[1], label_type="edge")
    #      plt.title("{} vs {}".format(correct_txt, wrong_txt))
    #      plt.xticks(rotation=60)
    #      plt.subplots_adjust(bottom=0.15)
    #      plt.tight_layout()
    #      fig = plot.get_figure()
    #      fig.savefig("{}/statistics_{}_{}.png".format(image_folder, statistics_name, i))
    #      plt.close()


def split(a: List[Any], n: int) -> List[Any]:
    """Split an array into equal intervals

    Args:
      a [List[Any]]: list
      n [int]: number of equal intervals

    Returns:
      list separaed with equal intervals
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def global_multiLabel_confusion_matrix(
    y_test_g: np.ndarray, y_est_g: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the multilabel confusion matrix

    Args:
      y_test [np.ndarray]: groundtruth
      y_est [np.ndarray]: test labels

    Returns:
      CM [np.ndarray]: confusion matrix
      Temp [np.ndarray]: temp variable
    """
    n_samples, n_class = y_test_g.shape
    CM = np.zeros((n_class, n_class))
    Temp = np.zeros((1, n_class))

    def acum_CM(y_test, y_est, CM, Temp):
        ind_real = np.asarray(y_test > 0).nonzero()[0]
        ind_est = np.asarray(y_est > 0).nonzero()[0]
        # --------------------------------
        if ind_real.size == 0:
            # In case in the ground truth not even one class is active
            Temp = Temp + y_est
        elif ind_est.size == 0:
            return CM, Temp
        else:
            mesh_real = np.array(np.meshgrid(ind_real, ind_real))
            comb_real = mesh_real.T.reshape(-1, 2)
            ind_remove_real = comb_real[:, 0] != comb_real[:, 1]
            comb_real = comb_real[ind_remove_real]
            # --------------------------------
            mesh_est = np.array(np.meshgrid(ind_real, ind_est))
            comb_est = mesh_est.T.reshape(-1, 2)
            # --------------------------------
            comb_real2 = comb_real[:, 0] + comb_real[:, 1] * 1j
            comb_est2 = comb_est[:, 0] + comb_est[:, 1] * 1j
            ind_remove = np.in1d(comb_est2, comb_real2)
            comb_est = comb_est[np.logical_not(ind_remove)]
            # --------------------------------
            CM[comb_est[:, 0], comb_est[:, 1]] += 1
        return CM, Temp

    for i in range(n_samples):
        CM, Temp = acum_CM(y_test_g[i, :], y_est_g[i, :], CM, Temp)

    return CM, Temp


def plot_global_multiLabel_confusion_matrix(
    y_test: np.ndarray,
    y_est: np.ndarray,
    label_names: str,
    normalize: bool,
    size: Tuple[int],
    fig_name: str,
) -> None:
    """Plot the confusion matrix in a global settings

    Args:
      y_test [np.ndarray]: groundtruth
      y_est [np.ndarray]: test labels
      normalize [bool]: whether to normalize the matrix
      size [Tuple[int]]: size of the figure
      fig_name [str]: name of the figure
    """
    fig, ax = plt.subplots(figsize=size)

    CM, Temp = global_multiLabel_confusion_matrix(y_test, y_est)

    #  # Normalization to show precision on the diagonal
    if normalize:
        CM = np.divide(CM, CM.sum(axis=0) + Temp, where=CM.sum(axis=0) + Temp != 0)

    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=label_names)
    #  print(CM)
    disp.plot(
        include_values=False,
        cmap="viridis",
        ax=ax,
        xticks_rotation="vertical",
        values_format=None,
        colorbar=True,
    )
    plt.title("Global multi-label confusion matrix")
    print(fig_name)
    fig.savefig("{}.png".format(fig_name))
    plt.close(fig)

    #  confusion_matrices = multilabel_confusion_matrix(y_test, y_est)
    #  for i, confusion_matrix_original in enumerate(confusion_matrices):
    #      for norm in ["true", "pred", "all", None]:
    #          confusion_matrix = confusion_matrix_original.copy()
    #
    #          # normalize
    #          if norm == "true":
    #              confusion_matrix = np.divide(
    #                  confusion_matrix,
    #                  confusion_matrix.sum(axis=1, keepdims=True),
    #                  where=confusion_matrix.sum(axis=1, keepdims=True) != 0,
    #              )
    #          elif norm == "pred":
    #              confusion_matrix = np.divide(
    #                  confusion_matrix,
    #                  confusion_matrix.sum(axis=0, keepdims=True),
    #                  where=confusion_matrix.sum(axis=0, keepdims=True) != 0,
    #              )
    #          elif norm == "all":
    #              confusion_matrix = np.divide(
    #                  confusion_matrix,
    #                  confusion_matrix.sum(),
    #                  where=confusion_matrix.sum() != 0,
    #              )
    #
    #          fig, ax = plt.subplots(figsize=(5, 5))
    #          disp = ConfusionMatrixDisplay(
    #              confusion_matrix,
    #              display_labels=["N", "Y"],
    #          )
    #          # "viridis"
    #          plt.title("Multi-label confusion matrix: {}\n\n".format(label_names[i]))
    #          plt.ylabel("True label")
    #          plt.xlabel("Predicted label")
    #          disp.plot(
    #              include_values=True,
    #              cmap=plt.cm.Blues,
    #              ax=ax,
    #              xticks_rotation="horizontal",
    #          )
    #          fig.savefig(
    #              "{}_{}{}.png".format(
    #                  fig_name,
    #                  label_names[i],
    #                  "" if norm is None else "_norm_{}".format(norm),
    #              ),
    #          )
    #          plt.close()


def plot_confusion_matrix_statistics(
    clf_report: Dict[str, float], fig_name: str
) -> None:
    """Plot the confusion matrix statistics

    Args:
      clf_report [Dict[str, float]]: statistics of the confusion matrix
      fig_name [str]: name of the figure
    """
    plt.figure(figsize=(15, 30))
    plt.title("Confusion matrix statistics")
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap=plt.cm.Blues)
    plt.savefig("{}".format(fig_name))
    plt.close()


def force_prediction_from_batch(
    output: torch.Tensor,
    prediction_treshold: float,
    use_softmax: bool,
    superclasses_number: int = 20,
) -> torch.Tensor:
    """Force the prediction from a batch of predictions

    Args:
      output [torch.Tensor]: output
      prediction_treshold [float]: threshold for the prediction
      use_softmax [bool] use softmax for the prediction
      superclasses_number [int] = 20: number of superclasses

    Returns:
      forced prediction [torch.Tensor]
    """
    new_output = list()
    for pred in output:
        if use_softmax:
            tmp = torch.zeros_like(pred, dtype=torch.bool)
            parent = torch.argmax(pred[1 : superclasses_number + 1]).item()
            child = torch.argmax(pred[superclasses_number + 1 :]).item()
            tmp = torch.zeros_like(pred, dtype=torch.bool)
            tmp[1 + parent] = True
            tmp[superclasses_number + 1 + child] = True
        else:
            tmp = pred > prediction_treshold
            if tmp[1:].sum().item() == 0:
                max_pred = torch.max(pred[1:]).item()
                tmp = pred >= max_pred
        new_output.append(tmp)
    return torch.stack(new_output)


def cross_entropy_from_softmax(
    targets: torch.Tensor, outputs: torch.Tensor, superclasses_number: int
) -> Tuple[float, float, float]:
    """Build the cross entropy from the softmax operation performed by the networks:

    Args:
        targets [torch.Tensor]: targets
        outputs [torch.Tensor]: outputs
        superclasses_number [int]: int
    """
    _, inds_0 = torch.max(targets[:, 1 : superclasses_number + 1], dim=1)
    loss_1 = F.nll_loss(torch.log(outputs[:, 1 : superclasses_number + 1]), inds_0)
    _, inds_1 = torch.max(targets[:, superclasses_number + 1 :], dim=1)
    loss_2 = F.nll_loss(torch.log(outputs[:, superclasses_number + 1 :]), inds_1)
    loss = loss_1 + loss_2
    return loss / 2, loss_1, loss_2


def get_confounders(dataset: str) -> Dict:
    """Given the dataset name, it returns the confounder
    Args:
        dataset [str]: name of the dataset
    Returns:
        confounders
    """
    confunders = cifar_confunders
    if dataset == "mnist":
        confunders = mnist_confunders
    elif dataset == "fashion":
        confunders = fashion_confunders
    elif dataset == "omniglot":
        confunders = omniglot_confunders
    return confunders


def get_hierarchy(hierarchy_name: str):
    """Given the hierarchy name, it returns the right hierarchy
    Args:
        hierarchy_name [str]: name of the hierarchy
    Returns:
        confounders
    """
    hierarchy = cifar_hierarchy
    if hierarchy_name == "mnist":
        hierarchy = mnist_hierarchy
    elif hierarchy_name == "fashion":
        hierarchy = fashion_hierarchy
    elif hierarchy_name == "omniglot":
        hierarchy = omniglot_hierarchy
    return hierarchy


def get_confounders_and_hierarchy(dataset: str):
    """Given the hierarchy name, it returns the right confounders and hierarchy
    Args:
        dataset [str]: name of the dataset
    Returns:
        confounders and hierarchy values and hierarchy keys
    """
    hierarchy = get_hierarchy(dataset)
    confounders = get_confounders(dataset)
    return confounders, hierarchy.values(), hierarchy.keys()


def prepare_dict_label_predictions_from_raw_predictions(
    predictions: np.ndarray,
    groundtruth: np.ndarray,
    label_names: List[str],
    dataset_name: str,
    skip_parents: bool,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Prepare dictionary label prediction

    Args:
        predictions [np.ndarray]: machine predictions
        groundtruth [np.ndarray]: groundtruth
        label_names [List[str]]: label names
        dataset_name [str]: dataset name
        skip_parents [bool]: skip the parents
    Returns:
        Tuple[Dict[str, Dict[str, int]], Dict[str, int]]
    """
    dictionary: Dict[str, Dict[str, int]] = {}
    counter: Dict[str, int] = {}
    _, _, parents = get_confounders_and_hierarchy(dataset_name)
    for item_ground, item_pred in zip(groundtruth, predictions):
        groundtruth_indexes = np.where(item_ground == 1)[0].tolist()
        parent, children = (
            label_names[groundtruth_indexes[0]],
            label_names[groundtruth_indexes[1]],
        )
        if not children in dictionary:
            dictionary[children] = {}
            counter[children] = 0
        counter[children] += 1

        prediction_indexes = np.where(item_pred == 1)[0].tolist()
        for idx in prediction_indexes:
            label_idx = label_names[idx]
            if skip_parents and label_idx in parents:
                continue
            if not label_idx in dictionary[children]:
                dictionary[children][label_idx] = 0
            dictionary[children][label_idx] += 1
    return (dictionary, counter)


def plot_confounded_labels_predictions(
    predictions: Dict[str, Dict[str, int]],
    counter: Dict[str, int],
    folder: str,
    prefix: str,
    dataset_name: str,
):
    """Plots a boxplot which depicts the predictions peformed over the data
    Args:
        predictions [Dict[str, Dict[str, int]]]: for each class to be predicted, which is the prediction
        and how many saples have been predicted
        folder [str]: folder
        prefix [str]: prefix
        dataset_name [str]: dtaset

    Returns:
        confounders and hierarchy values and hierarchy keys
    """
    # produce many plots
    for groundtruth_class in predictions:
        fig = plt.figure(figsize=(12, 8))
        data = [
            predictions[groundtruth_class][predicted_class]
            for predicted_class in predictions[groundtruth_class]
        ]
        titles = [predicted_class for predicted_class in predictions[groundtruth_class]]
        plt.bar(titles, data, color="blue", width=0.4)
        plt.title(
            "Barplot predictions for {} class #{}".format(
                groundtruth_class, counter[groundtruth_class]
            )
        )
        for index, d in enumerate(data):
            plt.text(x=index, y=d + 1, s=f"{d}")
        plt.xticks(rotation=0)
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        fig.savefig("{}/barplot_for_{}.png".format(folder, groundtruth_class))
        plt.close(fig)


###### Prepare probabilistic circuit #############


def prepare_probabilistic_circuit(
    A: torch.Tensor,
    constraint_folder: str,
    dataset_name: str,
    device: str,
    gates: int,
    num_reps: int,
    output_classes: int,
    S: int = 0,
) -> Tuple[CircuitMPE, DenseGatingFunction]:
    """Method which prepares an effective probabilistic circuit

    Args:
        A [torch.Tensor]: Adjency matrix
        constraint_folder [str]: constraint folder
        output_classes [int]: output classes
        dataset_name [str]: dataset name
        device [str]: device
        gates [int]: gates
        num_reps [int]: num reps
        output_classes [int]: output classes
        S [int] = 0: S

    Returns:
        Tuple[CircuitMPE, DenseGatingFunction]: circuit and gate
    """

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
    if (
        not os.path.isfile(constraint_folder + "/" + dataset_name + ".sdd")
        or not os.path.isfile(constraint_folder + "/" + dataset_name + ".vtree")
        or True
    ):
        # Compute matrix of ancestors R
        # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
        R = np.zeros(A.shape)
        np.fill_diagonal(R, 1)
        g = nx.DiGraph(A)
        for i in range(len(A)):
            descendants = list(nx.descendants(g, i))
            if descendants:
                R[i, descendants] = 1
        R = torch.tensor(R)

        # Transpose to get the ancestors for each node
        R = R.unsqueeze(0).to(device)

        # Uncomment below to compile the constraint
        R.squeeze_()

        # sdd manager
        mgr = SddManager(var_count=R.size(0), auto_gc_and_minimize=True)

        # Alpha represent our probabilistic circuit
        alpha = mgr.true()
        alpha.ref()

        # Creating the constraints
        for i in range(R.size(0)):
            beta = mgr.true()
            beta.ref()
            for j in range(R.size(0)):
                # if i is children of j and i is not j -> create the implication
                if R[i][j] and i != j:
                    old_beta = beta
                    # predict me and the other
                    beta = beta & mgr.vars[j + 1]
                    beta.ref()
                    old_beta.deref()

            # create the implication a -> b = not a or b
            old_beta = beta
            beta = -mgr.vars[i + 1] | beta
            beta.ref()
            old_beta.deref()

            old_alpha = alpha
            alpha = alpha & beta
            alpha.ref()
            old_alpha.deref()

        # Saving circuit & vtree to disk
        alpha.save(str.encode(constraint_folder + "/" + dataset_name + ".sdd"))
        alpha.vtree().save(
            str.encode(constraint_folder + "/" + dataset_name + ".vtree")
        )

    # Create circuit object
    cmpe = CircuitMPE(
        constraint_folder + "/" + dataset_name + ".vtree",
        constraint_folder + "/" + dataset_name + ".sdd",
    )

    # salvo qui come jason
    #  cmpe.rand_params()
    #  io.psdd_jason_save(cmpe.beta, constraint_folder + "/" + dataset_name + ".pysdd")
    #  exit(0)

    # overparameterization
    if S > 0:
        cmpe.overparameterize(S=S)
        print("Done overparameterizing")

    # Create gating function
    gate = DenseGatingFunction(
        cmpe.beta,
        gate_layers=[output_classes] + [output_classes] * gates,
        num_reps=num_reps,
        device=device,
    ).to(device)

    return (cmpe, gate)


def prepare_empty_probabilistic_circuit(
    A: torch.Tensor,
    device: str,
    output_classes: int,
) -> Tuple[CircuitMPE, DenseGatingFunction]:
    """Method which prepares an empty probabilistic circuit

    Args:
        A [torch.Tensor]: Adjency matrix
        device [str]: device
        output_classes [int]: output classes

    Returns:
        Tuple[CircuitMPE, DenseGatingFunction]: circuit and gate
    """

    print("Preparing empty circuit...")
    # Use fully-factorized sdd
    mgr = SddManager(var_count=A.shape[0], auto_gc_and_minimize=True)
    alpha = mgr.true()
    vtree = Vtree(var_count=A.shape[0], var_order=list(range(1, A.shape[0] + 1)))
    alpha.save(str.encode("ancestry.sdd"))
    vtree.save(str.encode("ancestry.vtree"))
    cmpe = CircuitMPE("ancestry.vtree", "ancestry.sdd")
    cmpe.overparameterize()

    # Gating function
    gate = DenseGatingFunction(
        cmpe.beta, gate_layers=[output_classes], device=device
    ).to(device)
    return (cmpe, gate)


def activate_dropout(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
