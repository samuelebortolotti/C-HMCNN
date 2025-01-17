"""Test module"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import tqdm
from chmncc.utils import (
    dotdict,
    force_prediction_from_batch,
    cross_entropy_from_softmax,
    activate_dropout,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    average_precision_score,
    hamming_loss,
    jaccard_score,
)

from chmncc.dataset import get_named_label_predictions

import numpy as np
import itertools

from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE


def tr_image(img: torch.Tensor) -> torch.Tensor:
    r"""
    Function which computes the average of the image, to better visualize it
    in Tensor board

    Args:
        img [torch.Tensor]: image

    Returns:
        image after the processing [torch.Tensor]
    """
    return (img + 1) / 2


def test_step(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cost_function: torch.nn.modules.loss.BCELoss,
    title: str,
    test: dotdict,
    device: str = "gpu",
    debug_mode: bool = False,
    print_me: bool = False,
    prediction_treshold: float = 0.5,
    force_prediction: bool = False,
    use_softmax: bool = False,
    superclasses_number: int = 20,
    montecarlo: bool = False,
) -> Tuple[float, float, float, float, Optional[float], Optional[float]]:
    r"""Test function for the network.
    It computes the accuracy together with the area under the precision-recall-curve as a metric

    Args:
        net [nn.Module] network
        test_loader [torch.utils.data.DataLoader] test data loader
        cost_function [torch.nn.modules.loss.BCELoss] binary cross entropy function
        title [str]: title of the experiment
        test [dotdict] test set dictionary
        device [str] = "gpu": device on which to run the experiment
        debug_mode [bool] = False: whether the test is done on the debug dataloader
        print_me [bool] = False: whether to print the data or not
        prediction_treshold [float]: threshold used to consider a class as predicted
        force_prediction [bool]: force prediction or not
        use_softmax [bool]: use the softmax
        superclasses_number [int]: number of superclasses
        montecarlo: bool = False: montecarlo evaluation

    Returns:
        cumulative_loss [float] loss on the test set [not used to train!]
        cumulative_accuracy [float] accuracy on the test set in percentage
        score [float] area under the precision-recall curve raw
        score [float] area under the precision-recall curve const
        rigth_answer_parent [Optional[float]] right answer for the parent
        rigth_answer_children [Optional[float]] right answer for the children
    """
    total = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    cumulative_loss_children = None
    cumulative_loss_parent = None

    # set the network to evaluation mode
    net.eval()

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, items in tqdm.tqdm(
            enumerate(itertools.islice(test_loader, 1, 5000)), desc=title
        ):
            if debug_mode:
                # debug dataloader
                (
                    inputs,  # image
                    superclass,  # string label
                    subclass,  # string label
                    targets,  # hierarchical label [that matrix of 1 hot encodings]
                    confunder_pos_1_x,  # int position
                    confunder_pos_1_y,  # int position
                    confunder_pos_2_x,  # int position
                    confunder_pos_2_y,  # int position
                    confunder_shape,  # dictionary containing informations
                ) = items
            else:
                # test dataloader
                (inputs, targets) = items

            # load data into device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # forward pass
            outputs = net(inputs.float())

            # force prediction
            if force_prediction:
                predicted = force_prediction_from_batch(
                    outputs.data, prediction_treshold, use_softmax, superclasses_number
                )
            else:
                predicted = outputs.data > prediction_treshold  # 0.5

            if print_me:
                torch.set_printoptions(profile="full")
                print("Predicted:")
                print(predicted, superclass, subclass)
                print("Groundtruth:")
                print(targets)

                import matplotlib.pyplot as plt

                for i in inputs:
                    plt.imshow(i.permute(1, 2, 0))
                    plt.show()

            # total
            total += targets.shape[0] * targets.shape[1]
            # total correct predictions
            cumulative_accuracy += (predicted == targets.byte()).sum().item()

            # loss computation
            if use_softmax:
                loss, loss_parent, loss_children = cross_entropy_from_softmax(
                    targets, outputs, superclasses_number
                )
            else:
                loss = cost_function(outputs.double(), targets)
                loss_parent, loss_children = None, None

            # sum up the losses
            if loss_parent is not None and loss_children is not None:
                if cumulative_loss_children is None or cumulative_loss_parent is None:
                    cumulative_loss_children = 0
                    cumulative_loss_parent = 0
                cumulative_loss_parent += loss_parent.item()
                cumulative_loss_children += loss_children.item()

            # fetch prediction and loss value
            cumulative_loss += (
                loss.item()
            )  # Note: the .item() is needed to extract scalars from tensors

            # if montecarlo
            predictions = []
            if montecarlo:
                activate_dropout(net)
                n_samples = 20  # number of montecarlo runs
                # number of montecarlo runs
                for _ in range(n_samples):
                    outputs = net(inputs.float())
                    # predicted
                    if force_prediction:
                        predicted = force_prediction_from_batch(
                            outputs.data,
                            prediction_treshold,
                            use_softmax,
                            superclasses_number,
                        )
                    else:
                        predicted = outputs.data > prediction_treshold  # 0.5
                    predictions.append(predicted.float())

                # Aggregate the predictions
                predictions = torch.stack(predictions)
                mean_predictions = predictions.mean(dim=0)
                final_predictions = (mean_predictions >= 0.5).float()
                # rename the prediction
                predicted = final_predictions

            # compute the au(prc)
            predicted = predicted.to("cpu")
            cpu_constrained_output = outputs.to("cpu")
            targets = targets.to("cpu")

            if batch_idx == 0:
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = targets
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, targets), dim=0)

            # TODO force exit
            if batch_idx == 200:
                break

    # average precision score raw
    score_raw = average_precision_score(
        y_test[:, test.to_eval], constr_test.data[:, test.to_eval], average="micro"
    )

    # average precision score
    score_const = average_precision_score(
        y_test[:, test.to_eval],
        predicted_test.data[:, test.to_eval].to(torch.float),
        average="micro",
    )

    return (
        cumulative_loss / len(test_loader),
        cumulative_accuracy / total * 100,
        score_raw,
        score_const,
        None
        if cumulative_loss_parent is None
        else cumulative_loss_parent / len(test_loader),
        None
        if cumulative_loss_children is None
        else cumulative_loss_children / len(test_loader),
    )


def test_step_with_prediction_statistics(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cost_function: torch.nn.modules.loss.BCELoss,
    nodes: List[str],
    title: str,
    test: dotdict,
    labels_name: List[str],
    device: str = "gpu",
    prediction_treshold: float = 0.5,
    force_prediction: bool = False,
    use_softmax: bool = False,
    superclasses_number: int = 20,
    montecarlo: bool = False,
) -> Tuple[
    float,
    float,
    float,
    float,
    Dict[str, List[int]],
    Dict[str, List[int]],
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    Optional[float],
    Optional[float],
]:
    r"""Test function for the network.
    It computes the accuracy together with the area under the precision-recall-curve as a metric
    and returns the statistics of predicted and non-predicted data.
    Predicted means that the network has put at least one value positive, whereas unpredicted means that all the predictions are zeros

    Args:
        net [nn.Module] network
        test_loader [torch.utils.data.DataLoader] test data loader with labels
        cost_function [torch.nn.modules.loss.BCELoss] binary cross entropy function
        title [str]: title of the experiment
        test [dotdict] test set dictionary
        labels_name [List[str]] name of the labels
        device [str] = "gpu": device on which to run the experiment
        prediction_treshold [float]: threshold used to consider a class as predicted
        force_prediction [bool]: force prediction
        use_softmax [bool]: use the softmax
        superclasses_number [int]: number of superclasses
        montecarlo: bool = False: montecarlo evaluation

    Returns:
        cumulative_loss [float] loss on the test set [not used to train!]
        cumulative_accuracy [float] accuracy on the test set in percentage
        score [float] area under the precision-recall curve raw
        score [float] area under the precision-recall curve const
        statistics [Dict[str, List[int]]]: name of the class : [not-predicted, predicted]
        statistics correct [Dict[str, List[int]]]: name of the class : [not-correct, correct]
        clf_report Dict[str, float]: confusion matrix statistics
        ground_truth ndarray: ground_truth predictions
        prediction ndarray: predictions
        rigth_answer_parent [Optional[float]] right answer for the parent
        rigth_answer_children [Optional[float]] right answer for the children
    """
    total = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    cumulative_loss_children = None
    cumulative_loss_parent = None

    # set the network to evaluation mode
    net.eval()

    # statistics
    stats_predicted = {"total": [0, 0]}
    stats_correct = {"total": [0, 0]}

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, items in tqdm.tqdm(
            enumerate(itertools.islice(test_loader, 1, 5000)), desc=title
        ):
            (inputs, superclass, subclass, targets) = items

            # load data into device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # forward pass
            outputs = net(inputs.float())

            # predicted
            if force_prediction:
                predicted = force_prediction_from_batch(
                    outputs.data, prediction_treshold, use_softmax, superclasses_number
                )
            else:
                predicted = outputs.data > prediction_treshold  # 0.5

            # total
            total += targets.shape[0] * targets.shape[1]
            # total correct predictions
            cumulative_accuracy += (predicted == targets.byte()).sum().item()

            # loss computation
            if use_softmax:
                loss, loss_parent, loss_children = cross_entropy_from_softmax(
                    targets, outputs, superclasses_number
                )
            else:
                loss = cost_function(outputs.double(), targets)
                loss_parent, loss_children = None, None

            # sum up the losses
            if loss_parent is not None and loss_children is not None:
                if cumulative_loss_children is None or cumulative_loss_parent is None:
                    cumulative_loss_children = 0
                    cumulative_loss_parent = 0
                cumulative_loss_parent += loss_parent.item()
                cumulative_loss_children += loss_children.item()

            # fetch prediction and loss value
            cumulative_loss += (
                loss.item()
            )  # Note: the .item() is needed to extract scalars from tensors

            # if montecarlo
            predictions = []
            if montecarlo:
                activate_dropout(net)
                n_samples = 20  # number of montecarlo runs
                # number of montecarlo runs
                for _ in range(n_samples):
                    outputs = net(inputs.float())
                    # predicted
                    if force_prediction:
                        predicted = force_prediction_from_batch(
                            outputs.data,
                            prediction_treshold,
                            use_softmax,
                            superclasses_number,
                        )
                    else:
                        predicted = outputs.data > prediction_treshold  # 0.5
                    predictions.append(predicted.float())

                # Aggregate the predictions
                predictions = torch.stack(predictions)
                mean_predictions = predictions.mean(dim=0)
                final_predictions = (mean_predictions >= 0.5).float()
                # rename the prediction
                predicted = final_predictions

            # compute the au(prc)
            predicted = predicted.to("cpu")
            cpu_constrained_output = outputs.to("cpu")
            targets = targets.to("cpu")

            if batch_idx == 0:
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = targets
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, targets), dim=0)

            # stats
            for i in range(inputs.shape[0]):
                # adding the predictions
                named_pred = get_named_label_predictions(predicted[i], nodes)
                predicted_idx = 1
                for pred in named_pred:
                    if not pred in stats_predicted:
                        stats_predicted[pred] = [0, 0]
                    stats_predicted[pred][predicted_idx] += 1
                stats_predicted["total"][predicted_idx] += 1

                # adding the not predicted
                predicted_idx = 0

                # already predicted
                if subclass[i] in named_pred and superclass[i] in named_pred:
                    continue
                # adding not predicted
                if not subclass[i] in stats_predicted:
                    stats_predicted[subclass[i]] = [0, 0]
                if not superclass[i] in stats_predicted:
                    stats_predicted[superclass[i]] = [0, 0]
                stats_predicted["total"][predicted_idx] += 1
                stats_predicted[superclass[i]][predicted_idx] += 1
                stats_predicted[subclass[i]][predicted_idx] += 1

            # create the statistics - correct
            for i in range(inputs.shape[0]):
                correct_idx = (
                    0
                    if not (predicted[i] == targets[i].byte()).sum()
                    == len(predicted[i])
                    else 1
                )

                if not subclass[i] in stats_correct:
                    stats_correct[subclass[i]] = [0, 0]
                if not superclass[i] in stats_correct:
                    stats_correct[superclass[i]] = [0, 0]
                stats_correct["total"][correct_idx] += 1
                stats_correct[superclass[i]][correct_idx] += 1
                stats_correct[subclass[i]][correct_idx] += 1

            # TODO force exit
            if batch_idx == 200:
                break

    # average precision score
    score_raw = average_precision_score(
        y_test[:, test.to_eval], constr_test.data[:, test.to_eval], average="micro"
    )

    score_const = average_precision_score(
        y_test[:, test.to_eval],
        predicted_test.data[:, test.to_eval].to(torch.float),
        average="micro",
    )

    # classification report on the confusion matrix
    clf_report = classification_report(
        y_test[:, test.to_eval],
        predicted_test[:, test.to_eval],
        output_dict=True,
        target_names=labels_name,
        zero_division=0,
    )

    return (
        cumulative_loss / len(test_loader),
        cumulative_accuracy / total * 100,
        score_raw,
        score_const,
        stats_predicted,
        stats_correct,
        clf_report,  # classification matrix
        y_test[:, test.to_eval],  # ground-truth for multiclass classification matrix
        predicted_test[
            :, test.to_eval
        ],  # predited values for multiclass classification matrix
        None
        if cumulative_loss_parent is None
        else cumulative_loss_parent / len(test_loader),
        None
        if cumulative_loss_children is None
        else cumulative_loss_children / len(test_loader),
    )


def test_circuit(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    title: str,
    test: dotdict,
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    device: str = "gpu",
    debug_mode: bool = False,
    montecarlo: bool = False,
) -> Tuple[float, float, float, float, float]:
    r"""Test function for the circuit.
    It computes the accuracy together with the area under the precision-recall-curve, jaccard score and hamming score as a metric

    Args:
        net [nn.Module]: network
        test_loader [torch.utils.data.DataLoader]: test loader
        title [str]: title
        test [dotdict]: test
        gate [DenseGatingFunction]: gate
        cmpe [CircuitMPE]: circuit MPE
        device [str] = "gpu": device
        debug_mode [bool] = False: whether the debug mode is set
        montecarlo: bool = False: montecarlo evaluation

    Returns:
        cumulative_loss [float]: cumulative loss
        accuracy [float]: accuracy
        jaccard [float]: jaccard score
        hamming [float]: hamming score
        auprc_score [float]: area under precision and recall curve
    """
    cumulative_accuracy = 0.0
    cumulative_loss = 0.0
    total_train = 0

    # set the network to evaluation mode
    net.eval()
    gate.eval()

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, items in tqdm.tqdm(
            enumerate(itertools.islice(test_loader, 1, 5000)), desc=title
        ):
            if debug_mode:
                # debug dataloader
                (
                    inputs,  # image
                    superclass,  # string label
                    subclass,  # string label
                    targets,  # hierarchical label [that matrix of 1 hot encodings]
                    confunder_pos_1_x,  # int position
                    confunder_pos_1_y,  # int position
                    confunder_pos_2_x,  # int position
                    confunder_pos_2_y,  # int position
                    confunder_shape,  # dictionary containing informations
                ) = items
            else:
                # test dataloader
                (inputs, targets) = items

            # load data into device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            outputs = net(inputs.float())

            # thetas
            thetas = gate(outputs.float())

            # negative log likelihood and map
            cmpe.set_params(thetas)
            predicted = (cmpe.get_mpe_inst(inputs.shape[0]) > 0).long()

            # compute the loss
            cmpe.set_params(thetas)
            loss = cmpe.cross_entropy(
                targets, log_space=True
            ).mean()  # set them to None

            # if montecarlo
            predictions = []
            if montecarlo:
                activate_dropout(net)
                n_samples = 20  # number of montecarlo runs
                # number of montecarlo runs
                for _ in range(n_samples):
                    outputs = net(inputs.float())
                    thetas = gate(outputs.float())
                    cmpe.set_params(thetas)
                    predicted = (cmpe.get_mpe_inst(inputs.shape[0]) > 0).long()
                    predictions.append(predicted.float())
                # Aggregate the predictions
                predictions = torch.stack(predictions)
                mean_predictions = predictions.mean(dim=0)
                final_predictions = (mean_predictions >= 0.5).float()
                # rename the prediction
                predicted = final_predictions

            total_train += targets.shape[0] * targets.shape[1]
            num_correct = (predicted == targets.byte()).sum().item()

            # total correct predictions
            cumulative_accuracy += num_correct
            # total loss
            cumulative_loss += loss.item()

            # compute the au(prc)
            targets = targets.to("cpu")
            predicted = predicted.to("cpu")

            if batch_idx == 0:
                predicted_test = predicted
                y_test = targets
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                y_test = torch.cat((y_test, targets), dim=0)

            if batch_idx == 200:
                break

    y_test = y_test[:, test.to_eval]
    predicted_test = predicted_test[:, test.to_eval]

    # average precision score and other statistics
    accuracy = cumulative_accuracy / total_train
    jaccard = jaccard_score(y_test, predicted_test, average="micro")
    hamming = hamming_loss(y_test, predicted_test)
    auprc_score = average_precision_score(y_test, predicted_test, average="micro")

    return (
        cumulative_loss / (batch_idx + 1),
        accuracy * 100,
        jaccard,
        hamming,
        auprc_score,
    )


def test_step_with_prediction_statistics_with_gates(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    nodes: List[str],
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    title: str,
    test: dotdict,
    labels_name: List[str],
    device: str = "gpu",
    montecarlo: bool = False,
) -> Tuple[
    float,
    float,
    float,
    float,
    Dict[str, List[int]],
    Dict[str, List[int]],
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    float,
]:
    r"""Test function for the network with the circuit.
    It computes the accuracy together with the area under the precision-recall-curve, jaccard score and hamming loss as a metric
    and returns the statistics of predicted and non-predicted data.
    Predicted means that the network has put at least one value positive, whereas unpredicted means that all the predictions are zeros

    Args:
        net [nn.Module]: network
        test_loader [torch.utils.data.DataLoader]: test data loader with labels
        gate [DenseGatingFunction]: gate
        cmpe [CircuitMPE]: Circuit MPE
        title [str]: title
        test [dotdict]: test
        labels_name [List[str]: labels name
        device [str] = "gpu": device
        montecarlo [bool] = False: montecarlo approccio

    Returns:
        cumulative_loss [float] loss on the test set [not used to train!]
        cumulative_accuracy [float] accuracy on the test set in percentage
        jaccard score [float]
        hamming loss [float]
        statistics [Dict[str, List[int]]]: name of the class : [not-predicted, predicted]
        statistics correct [Dict[str, List[int]]]: name of the class : [not-correct, correct]
        clf_report Dict[str, float]: confusion matrix statistics
        y_test [np.ndarray]: ground-truth for multiclass classification matrix
        predicted_test [ndarray]: predictions
        auprc_score [float] auprc score
    """
    cumulative_accuracy = 0.0
    cumulative_loss = 0.0
    total_train = 0

    # statistics
    stats_predicted = {"total": [0, 0]}
    stats_correct = {"total": [0, 0]}

    # set the network to evaluation mode
    net.eval()
    gate.eval()

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, items in tqdm.tqdm(
            enumerate(itertools.islice(test_loader, 1, 5000)), desc=title
        ):
            (inputs, superclass, subclass, targets) = items

            # load data into device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # forward pass
            outputs = net(inputs.float())
            # thetas
            thetas = gate(outputs.float())

            # negative log likelihood and map
            cmpe.set_params(thetas)
            predicted = (cmpe.get_mpe_inst(inputs.shape[0]) > 0).long()

            # compute the loss
            cmpe.set_params(thetas)
            loss = cmpe.cross_entropy(
                targets, log_space=True
            ).mean()  # set them to None

            # if montecarlo
            predictions = []
            if montecarlo:
                activate_dropout(net)
                n_samples = 20  # number of montecarlo runs
                # number of montecarlo runs
                for _ in range(n_samples):
                    outputs = net(inputs.float())
                    thetas = gate(outputs.float())
                    cmpe.set_params(thetas)
                    predicted = (cmpe.get_mpe_inst(inputs.shape[0]) > 0).long()
                    predictions.append(predicted.float())
                # Aggregate the predictions
                predictions = torch.stack(predictions)
                mean_predictions = predictions.mean(dim=0)
                final_predictions = (mean_predictions >= 0.5).float()
                # rename the prediction
                predicted = final_predictions

            total_train += targets.shape[0] * targets.shape[1]
            num_correct = (predicted == targets.byte()).sum().item()

            # total correct predictions
            cumulative_accuracy += num_correct
            # total loss
            cumulative_loss += loss.item()

            # compute the au(prc)
            targets = targets.to("cpu")
            predicted = predicted.to("cpu")

            print("In test")
            print(predicted)
            print(targets)
            print(num_correct)
            print("--------------------------")

            if batch_idx == 0:
                predicted_test = predicted
                print()
                print("Ci entro")
                print()
                y_test = targets
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                y_test = torch.cat((y_test, targets), dim=0)

            # stats
            for i in range(inputs.shape[0]):
                # adding the predictions
                named_pred = get_named_label_predictions(predicted[i], nodes)
                predicted_idx = 1
                for pred in named_pred:
                    if not pred in stats_predicted:
                        stats_predicted[pred] = [0, 0]
                    stats_predicted[pred][predicted_idx] += 1
                stats_predicted["total"][predicted_idx] += 1

                # adding the not predicted
                predicted_idx = 0

                # already predicted
                if subclass[i] in named_pred and superclass[i] in named_pred:
                    continue
                # adding not predicted
                if not subclass[i] in stats_predicted:
                    stats_predicted[subclass[i]] = [0, 0]
                if not superclass[i] in stats_predicted:
                    stats_predicted[superclass[i]] = [0, 0]
                stats_predicted["total"][predicted_idx] += 1
                stats_predicted[superclass[i]][predicted_idx] += 1
                stats_predicted[subclass[i]][predicted_idx] += 1

            # create the statistics - correct
            for i in range(inputs.shape[0]):
                correct_idx = (
                    0
                    if not (predicted[i] == targets[i].byte()).sum()
                    == len(predicted[i])
                    else 1
                )

                if not subclass[i] in stats_correct:
                    stats_correct[subclass[i]] = [0, 0]
                if not superclass[i] in stats_correct:
                    stats_correct[superclass[i]] = [0, 0]
                stats_correct["total"][correct_idx] += 1
                stats_correct[superclass[i]][correct_idx] += 1
                stats_correct[subclass[i]][correct_idx] += 1

            if batch_idx == 200:
                break

    y_test = y_test[:, test.to_eval]
    predicted_test = predicted_test[:, test.to_eval]

    # average precision score and other statistics
    accuracy = cumulative_accuracy / total_train
    jaccard = jaccard_score(y_test, predicted_test, average="micro")
    hamming = hamming_loss(y_test, predicted_test)
    auprc_score = average_precision_score(y_test, predicted_test, average="micro")

    # classification report on the confusion matrix
    clf_report = classification_report(
        y_test,
        predicted_test,
        output_dict=True,
        target_names=labels_name,
        zero_division=0,
    )

    return (
        cumulative_loss / (batch_idx + 1),
        accuracy * 100,
        jaccard,
        hamming,
        stats_predicted,
        stats_correct,
        clf_report,  # classification matrix
        y_test,  # ground-truth for multiclass classification matrix
        predicted_test,  # predited values for multiclass classification matrix
        auprc_score,
    )
