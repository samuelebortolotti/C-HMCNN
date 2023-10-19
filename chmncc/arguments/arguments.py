"""Analyze arguments"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
import seaborn as sns
import os
import torch.nn as nn
import scipy
import itertools
from collections import Counter
import pandas as pd
from chmncc.networks import ResNet18, LeNet5, LeNet7, AlexNet, MLP
from chmncc.utils.utils import (
    force_prediction_from_batch,
    load_best_weights,
    load_last_weights,
    load_best_weights_gate,
    dotdict,
    split,
    get_confounders,
    prepare_probabilistic_circuit,
    prepare_empty_probabilistic_circuit,
)
from chmncc.dataset import (
    load_dataloaders,
    get_named_label_predictions,
)
from typing import Dict, Any, Tuple, List
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torchsummary import summary
from sklearn.linear_model import RidgeClassifier
from chmncc.arguments.arguments_bucket import ArgumentBucket
from chmncc.debug.debug import plot_decision_surface

from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE


class ArgumentsStepArgs:
    """Class which serves as a bucket in order to hold all the data used so as to produce the plots"""

    """List of the bucket associated to the analyzed samples"""
    bucket_list: List[ArgumentBucket]
    """
    Table correlation dictionary for each class which contains a dictionary for each arguments name
    which contains a tuple representing whether the same has been correcly guessed and the score associated
    """
    table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]
    """List of input gradients explainations which is taken from the explainations which should be the 'right' ones"""
    suitable_ig_explaination_list: List[float]
    """Lists of gradients for each sample, where the first list is the list of input gradients while the second list concerns the label gradient. This concerns only the 'right' classes"""
    suitable_gradient_full_list: List[Tuple[List[float], List[float]]]
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the input gradient]"""
    max_arguments_list: List[float]
    """Lists of gradients for each sample, where the first list is the list of input gradients while the second list concerns the label gradient"""
    ig_lists: List[Tuple[List[float], List[float]]]
    """Max arguments dictionary"""
    max_arguments_dict: Dict[str, Dict[str, float]]
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the label gradient]"""
    label_args_dict: Dict[str, Dict[str, List[float]]]
    """Counter for showing the single element example"""
    show_element_counter: int
    """Counter which defines how many times a subclass has influenced a superclass"""
    influence_parent_counter: int
    """Counter which defines how many times a subclass has not influenced a superclass"""
    not_influence_parent_counter: int
    """Counter which defined how many times the 'right' argument is the maximum one"""
    suitable_is_max: int
    """Counter which defined how many times the 'right' argument is not the maximum one"""
    suitable_is_not_max: int
    """Entropy of the predicted label for each sample"""
    label_entropy_list: List[float]
    """Entropy map for input gradient"""
    ig_entropy_list: List[float]

    def __init__(
        self,
        bucket_list: List[ArgumentBucket],
        table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]],
        suitable_ig_explaination_list: List[float],
        suitable_gradient_full_list: List[Tuple[List[float], List[float]]],
        ig_lists: List[Tuple[List[float], List[float]]],
        max_arguments_list: List[float],
        max_ig_label_list_for_score_plot: List[Tuple[List[float], List[float]]],
        max_arguments_dict: Dict[str, Dict[str, float]],
        label_args_dict: Dict[str, Dict[str, List[float]]],
        show_element_counter: int,
        influence_parent_counter: int,
        not_influence_parent_counter: int,
        suitable_is_max: int,
        suitable_is_not_max: int,
        prediction_influence_parent_counter: int,
        prediction_does_not_influence_parent_counter: int,
        ig_lists_wrt_prediction: List[Tuple[List[float], List[float]]],
        label_entropy_list: List[float],
        ig_entropy_list: List[float],
    ):
        self.bucket_list = bucket_list
        self.table_correlation = table_correlation
        self.suitable_ig_explaination_list = suitable_ig_explaination_list
        self.suitable_gradient_full_list = suitable_gradient_full_list
        self.max_ig_label_list_for_score_plot = max_ig_label_list_for_score_plot
        self.ig_lists = ig_lists
        self.max_arguments_list = max_arguments_list
        self.max_arguments_dict = max_arguments_dict
        self.label_args_dict = label_args_dict
        self.show_element_counter = show_element_counter
        self.influence_parent_counter = influence_parent_counter
        self.not_influence_parent_counter = not_influence_parent_counter
        self.suitable_is_max = suitable_is_max
        self.suitable_is_not_max = suitable_is_not_max
        self.prediction_influence_parent_counter = prediction_influence_parent_counter
        self.prediction_does_not_influence_parent_counter = (
            prediction_does_not_influence_parent_counter
        )
        self.ig_lists_wrt_prediction = ig_lists_wrt_prediction
        self.label_entropy_list = label_entropy_list
        self.ig_entropy_list = ig_entropy_list


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the debug of the network
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("arguments", help="Analyze network arguments")
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["resnet", "lenet", "lenet7", "alexnet", "mlp"],
        default="resnet",
        help="Network",
    )
    parser.add_argument("--iterations", "-it", type=int, default=30, help="Debug Epocs")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--weights-path-folder",
        "-wpf",
        type=str,
        default="models",
        help="Path where to load the best.pth file",
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="Train batch size"
    )
    parser.add_argument(
        "--test-batch-size", "-tbs", type=int, default=128, help="Test batch size"
    )
    parser.add_argument(
        "--arguments-folder",
        "-af",
        type=str,
        default="arguments",
        help="Arguments folder",
    )
    parser.add_argument(
        "--model-folder",
        "-mf",
        type=str,
        default="models",
        help="Folder where to save the model",
    )
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
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device",
    )
    parser.add_argument(
        "--prediction-treshold",
        type=float,
        default=0.5,
        help="considers the class to be predicted in a multilabel classification setting",
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
        "--dataset",
        type=str,
        default="cifar",
        choices=["mnist", "cifar", "fashion", "omniglot"],
        help="which dataset to use",
    )
    parser.add_argument(
        "--simplified-dataset",
        "-simdat",
        dest="simplified_dataset",
        action="store_true",
        help="If possibile, use a simplified version of the dataset",
    )
    parser.add_argument(
        "--num-element-to-analyze",
        "-neta",
        dest="num_element_to_analyze",
        type=int,
        help="Number of elements to consider",
        default=2,
    )
    parser.add_argument(
        "--norm-exponent",
        "-nexp",
        dest="norm_exponent",
        type=int,
        help="Norm exponent",
        default=2,
    )
    parser.add_argument(
        "--tau",
        "-t",
        dest="tau",
        type=float,
        help="Tau for gradient analysis table",
        default=0.5,
    )
    parser.add_argument(
        "--multiply-by-probability-for-label-gradient",
        "-mbpflg",
        action="store_true",
        help="Use the probability of the node so as to compute the label gradient score",
    )
    parser.add_argument(
        "--cincer-approach",
        "-cincer",
        action="store_true",
        dest="cincer",
        help="Use the cincer approach in order to compute the label gradient",
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
        help="Number of hidden layers in gating function (default: 1)",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=1,
        help="Number of hidden layers in gating function (default: 1)",
    )
    parser.add_argument(
        "--S", type=int, default=0, help="PSDD scaling factor (default: 0)"
    )
    parser.add_argument(
        "--constraint-folder",
        type=str,
        default="./constraints",
        help="Folder for storing the constraints",
    )
    parser.add_argument(
        "--use-gate-output",
        "-ugout",
        action="store_true",
        help="Whether to use the gate output",
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
        multiply_by_probability_for_label_gradient=False,
        cincer=False,
        use_gate_output=False,
    )


def single_element_barplot(
    class_name: str,
    idx: int,
    ig_list: List[float],
    ig_titles: List[str],
    arguments_folder: str,
) -> None:
    """Method which plots the gradient barplots for each of the arguments of a single sample
    Args:
        class_name [str]: class name [groundtruth label]
        idx [int]: index of the element
        ig_list [List[float]]: gradient list
        ig_titles [List[str]]: list of x label titles
        arguments_folder [str]: arguments folder
    """
    for i, (data, title) in enumerate(zip(split(ig_list, 10), split(ig_titles, 10))):
        fig = plt.figure(figsize=(8, 4))
        plt.bar(title, data, color="blue", width=0.4)
        plt.title("Gradients value: {}-{}".format(class_name, idx))
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        fig.savefig("{}/gradients_{}_{}.pdf".format(arguments_folder, str(idx), i))
        plt.close(fig)


def score_barplot(
    x1: int,
    x2: int,
    label_1: str,
    label_2: str,
    folder: str,
    title: str,
) -> None:
    """Method which plots a bar plot with two bars
    Args:
        x1 [int]: value of the first bar
        x2 [int]: value of the second bar
        label_1 [str]: label for the first bar
        label_2 [str]: label for the second bar
        folder [str]: folder
        title [str]: title
    """
    fig = plt.figure(figsize=(10, 6))
    titles = np.array([label_1, label_2])
    values = np.array([x1, x2])
    colors = sns.color_palette('Set2', n_colors=2)
    plot = pd.Series(values).plot(kind="bar", color=colors, edgecolor='black')
    plot.bar_label(plot.containers[0], label_type="edge", fontsize=10, padding=3, fmt='%.2f')
    plot.set_xticklabels(titles, fontsize=12)
    plt.xticks(rotation=0)
    plt.title(title, fontsize=16)
    plt.ylabel('Values', fontsize=12)
    plt.tight_layout()
    fig.savefig("{}/{}.pdf".format(folder, title))
    plt.close(fig)


def score_barplot_list(
    x1: List[int],
    x2: List[int],
    label_1: List[str],
    label_2: List[str],
    folder: str,
    title: str,
) -> None:
    """Method which plots a bar plot with multiple bars based on how many values are present withing the lists
    Args:
        x1 [int]: list of green values of the plot
        x2 [int]: list of red values of the plot
        label_1 [str]: list of label for the green part of the plot
        label_2 [str]: list of label for the second bar of the plot
        folder [str]: folder name of the plot
        title [str]: title of the plot
    """
    x_full = []
    color = []
    labels = []
    for el_1, el_2, el_lab_1, el_lab_2 in zip(x1, x2, label_1, label_2):
        x_full.append(el_1)
        x_full.append(el_2)
        labels.append(el_lab_1)
        labels.append(el_lab_2)
        color.append("green")
        color.append("red")

    fig = plt.figure(figsize=(10, 9))
    titles = np.array(labels)
    values = np.array(x_full)
    colors = sns.color_palette('Set2', n_colors=len(titles))

    plot = pd.Series(values).plot(kind="bar", color=colors, edgecolor='black')
    plot.bar_label(plot.containers[0], label_type="edge", fontsize=10, padding=3, fmt='%.2f')
    plot.set_xticklabels(titles, fontsize=12)
    plt.xticks(rotation="vertical")
    plt.title(title, fontsize=16)
    plt.ylabel('Values', fontsize=12)
    plt.tight_layout()
    fig.savefig("{}/{}.pdf".format(folder, title))
    plt.close(fig)


def input_gradient_scatter(
    correct_guesses: List[float],
    correct_guesses_conf: List[bool],
    wrongly_guesses: List[float],
    wrongly_guesses_conf: List[bool],
    folder: str,
    prefix: str,
) -> None:
    """Method which plots the scatter plot between the scores of confounded samples and not confounded samples
    Args:
        correct_guesses List[float]: list of scores associated with not confounded classes
        correct_guesses_conf: List[bool]: list of boolean values which represent whether the class is confounded or not
        wrongly_guesses List[float]: list of scores associated confounded classes
        wrongly_guesses_conf List[bool]: list of boolean values which represent whether the class is confounded or not
        folder [str]: folder name
        prefix [str]: prefix of the image name
    """
    # get all the score and correct list
    score_list = list(itertools.chain(correct_guesses, wrongly_guesses))
    correct_list = list(itertools.chain(correct_guesses_conf, wrongly_guesses_conf))

    # compute the correlation
    corr = scipy.stats.spearmanr(score_list, correct_list)

    # stacking the gradients with the x position => 0
    gradients_magnitude_sequence = np.array(score_list).reshape(-1, 1)
    gradients_magnitude_sequence_filled_x = np.column_stack(
        (
            np.zeros(gradients_magnitude_sequence.shape[0]),
            gradients_magnitude_sequence,
        )
    )
    counfounded_sequence = np.array(correct_list)

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
        "Score: {}, spearman correlation {:.3f}, p-val {:.3f}\nsignificant with 95% confidence {} #conf {} #not-conf {}".format(
            score,
            corr[0],
            corr[1],
            corr[1] < 0.05,
            len(wrongly_guesses),
            len(correct_guesses),
        ),
        True,
        0.001,
        max(score_list) + 0.1,
        False,
    )
    fig.savefig(
        "{}/{}_integrated_gradient_correlation.pdf".format(
            folder,
            prefix,
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
    plt.xticks([1, 2], ["Correct (not conf)", "Wrong (conf)"])

    for line in bp_dict["medians"]:
        # get position data for median line
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        plt.text(x, y, "%.5f" % y, horizontalalignment="center")

    fig.suptitle(
        "Boxplot: #conf {} #not-conf {}".format(
            len(wrongly_guesses), len(correct_guesses)
        )
    )
    fig.savefig(
        "{}/{}_integrated_gradient_boxplot.pdf".format(
            folder,
            prefix,
        )
    )
    plt.close(fig)


def display_single_class(
    class_name: str,
    gradient: torch.Tensor,
    score: float,
    single_el: torch.Tensor,
    predicted: str,
    arguments_folder: str,
    idx: int,
    prefix: str,
    correct: bool,
) -> None:
    """Overlay the input gradient over the base grayscale image

    Args:
        class_name [str]: groundtruth name of the class
        gradient [torch.Tensor]: gradient to show
        score [float]: score
        single_el [torch.Tensor]: element to display
        predicted [str]: predicted label
        arguments_folder [str]: arguments folder
        idx [int]: index of the element
        prefix [str]: prefix of the image
        correct [bool] correct prediction
    """
    gradient_to_show = np.fabs(gradient.detach().numpy().transpose(1, 2, 0))
    single_el = single_el.detach().numpy().transpose(1, 2, 0)

    # norm color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(gradient_to_show))
    # show the picture
    fig = plt.figure()
    plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
        label="Gradient magnitude",
    )
    plt.imshow(single_el, cmap="gray")
    plt.imshow(gradient_to_show, cmap="viridis", alpha=0.5)
    plt.subplots_adjust(top=0.72)
    plt.title(
        "\nInput gradient overlay wrt to {}-{}\nPredicted as {}\nScore: {}".format(
            class_name, predicted, idx, score
        )
    )

    # show the figure
    fig.savefig(
        "{}/{}_{}_input_gradient_{}_{}.pdf".format(
            arguments_folder,
            prefix,
            class_name,
            idx,
            "_correct" if correct else "",
        ),
        dpi=fig.dpi,
    )
    print(
        "Saving {}/{}_{}_input_gradient_{}_{}.pdf".format(
            arguments_folder,
            class_name,
            prefix,
            idx,
            "_correct" if correct else "input",
        )
    )
    plt.close(fig)


def arguments(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    arguments_folder: str,
    iterations: int,
    device: str,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    dataset: str,
    num_element_to_analyze: int,
    norm_exponent: int,
    tau: float,
    multiply_by_probability_for_label_gradient: bool,
    cincer: bool,
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    use_probabilistic_circuits: bool,
    use_gate_output: bool,
    **kwargs: Any,
) -> None:
    """Arguments method: it displays some functions useful in order to understand whether the score is suitable for the identification of the
    arguments.

    Args:
        net [nn.Module]: network
        dataloaders Dict[str, Any]: dataloaders
        arguments_folder [str]: arguments folder
        iterations [int]: interactions
        device [str]: device
        prediction_treshold [float]: prediction threshold
        force_prediction [bool]: force prediction
        use_softmax [bool]: whether to use softmax
        dataset [str] name of the dataset
        num_element_to_analyze [int]: number of elemenets to analyze
        norm_exponent [int] exponent to use in the norm computation
        **kwargs [Any]: kwargs
    """

    # labels name
    labels_name = dataloaders["test_set"].nodes_names_without_root

    print("Have to run for {} arguments iterations...".format(iterations))

    # running for the requested iterations
    for it in range(iterations):

        print("Start iteration number {}".format(it))
        print("-----------------------------------------------------")

        correct_confound = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=True,
            confounded_samples_only=True,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=False,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("Corr+Conf #", len(correct_confound.label_entropy_list))

        wrong_confound = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=False,
            confounded_samples_only=True,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=False,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("Wrong+Conf #", len(wrong_confound.label_entropy_list))

        correct_not_confound = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=True,
            confounded_samples_only=False,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=False,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("Corr+NotConf #", len(correct_not_confound.label_entropy_list))

        wrong_not_confound = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=False,
            confounded_samples_only=False,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=False,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("NotCorr+NotConf #", len(wrong_not_confound.label_entropy_list))

        correct_lab = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=True,
            confounded_samples_only=False,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=True,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("Corr+Imbalance #", len(correct_lab.label_entropy_list))

        wrong_lab = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=False,
            confounded_samples_only=False,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=True,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("NotCorr+Imbalance #", len(wrong_lab.label_entropy_list))

        # cases of both labels and image confounders
        wrong_lab_img = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=False,
            confounded_samples_only=True,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=True,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("NotCorr+Conf+Imbalance #", len(wrong_lab_img.label_entropy_list))

        corr_lab_img = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=True,
            confounded_samples_only=True,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=True,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("Corr+Conf+Imbalance #", len(corr_lab_img.label_entropy_list))

        corr_no_conf = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=True,
            confounded_samples_only=False,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=False,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("Corr+NotConf #", len(corr_no_conf.label_entropy_list))


        wrong_no_conf = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=False,
            confounded_samples_only=False,
            num_element_to_analyze=num_element_to_analyze,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder,
            label_loader=False,
            norm_exponent=norm_exponent,
            multiply_by_probability_for_label_gradient=multiply_by_probability_for_label_gradient,
            cincer=cincer,
            gate=gate,
            cmpe=cmpe,
            use_probabilistic_circuits=use_probabilistic_circuits,
            use_gate_output=use_gate_output,
        )
        print("NotCorr+NotConf #", len(wrong_no_conf.label_entropy_list))

        plot_arguments(
            correct_confound,
            wrong_confound,
            correct_not_confound,
            wrong_not_confound,
            correct_lab,
            wrong_lab,
            wrong_lab_img,
            corr_lab_img,
            corr_no_conf,
            wrong_no_conf,
            arguments_folder=arguments_folder,
            tau=tau,
        )


def get_table_correlation_dictionary_from(
    bucket: ArgumentBucket,
    correct_samples_only: bool,
    table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]],
) -> Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]:
    """Get the table correlation dictionary from the Bucket
    Args:
        bucket [ArgumentBucket]: sample argument bucket
        correct_samples_only [bool]: whether the examples are correctly guessed or not
        table_correlation Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]: table correlation
    Returns:
        Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]: dictionary for each class which contains a dictionary for each arguments name
        which contains a tuple representing whether the same has been correcly guessed and the score associated
    """
    # add the dictionary for each children class
    if not bucket.groundtruth_children in table_correlation:
        table_correlation[bucket.groundtruth_children] = {}
    dict_list = bucket.get_gradients_by_names()
    for key, value in dict_list.items():
        if not key in table_correlation[bucket.groundtruth_children]:
            table_correlation[bucket.groundtruth_children][key] = list()
        table_correlation[bucket.groundtruth_children][key].append(
            (correct_samples_only, value)
        )
    return table_correlation


def organize_aucs(
    correct_confound: ArgumentsStepArgs,
    wrong_confound: ArgumentsStepArgs,
    correct_not_confound: ArgumentsStepArgs,
    wrong_not_confound: ArgumentsStepArgs,
    correct_lab: ArgumentsStepArgs,
    wrong_lab: ArgumentsStepArgs,
    wrong_lab_img_confound: ArgumentsStepArgs,
    correct_lab_img_confound: ArgumentsStepArgs,
    corr_no_conf: ArgumentsStepArgs,
    wrong_no_conf: ArgumentsStepArgs,
    arguments_folder: str,
    num_values: int = 300,
    index_to_exclude: List[int] = []
):
    def return_split(
        correct_confound: ArgumentsStepArgs,
        wrong_confound: ArgumentsStepArgs,
        correct_not_confound: ArgumentsStepArgs,
        wrong_not_confound: ArgumentsStepArgs,
        correct_lab: ArgumentsStepArgs,
        wrong_lab: ArgumentsStepArgs,
        wrong_lab_img_confound: ArgumentsStepArgs,
        correct_lab_img_confound: ArgumentsStepArgs,
        corr_no_conf: ArgumentsStepArgs,
        wrong_no_conf: ArgumentsStepArgs,
        first_list_index: int,
        second_list_index: int,
        num_values: int = 300,
        index_to_exclude: List[int] = []
    ):
        # Combine all lists
        all_lists = [
            correct_confound.bucket_list,
            wrong_confound.bucket_list,
            correct_not_confound.bucket_list,
            wrong_not_confound.bucket_list,
            correct_lab.bucket_list,
            wrong_lab.bucket_list,
            wrong_lab_img_confound.bucket_list,
            correct_lab_img_confound.bucket_list,
            corr_no_conf.bucket_list,
            wrong_no_conf.bucket_list
        ]

        max_elements_from_first = 0
        if first_list_index not in index_to_exclude:
            max_elements_from_first = int(min(len(all_lists[first_list_index]), num_values/4))
        elements_from_second = 0
        if second_list_index not in index_to_exclude:
            elements_from_second = int(num_values/2 - max_elements_from_first)

        # Split exactly max_elements_from_first items from the first list
        first_elements = all_lists[first_list_index][:max_elements_from_first]
        # Split remaining elements from the second list
        second_elements = all_lists[second_list_index][:elements_from_second]
        print("First and second", len(first_elements), len(second_elements))

        # Exclude index 
        exclude_index = [first_list_index, second_list_index]
        exclude_index.extend(index_to_exclude)
        # Exclude the indices of the first and second lists
        remaining_indices = [i for i in range(len(all_lists)) if i not in exclude_index]
        print("Remaining indices", remaining_indices)
        # Calculate the number of remaining elements needed to reach a total of 300
        remaining_elements_needed = num_values/2
        elements_per_list = remaining_elements_needed // len(all_lists)
        # Split remaining elements from the remaining lists
        remaining_elements = []
        for i in remaining_indices:
            elements_to_add = int(min(len(all_lists[i]), elements_per_list))
            remaining_elements.extend(all_lists[i][:elements_to_add])
            remaining_elements_needed -= elements_to_add
        # If there are still elements needed, add them from the lists in a round-robin manner
        while remaining_elements_needed > 0:
            for i in remaining_indices:
                if remaining_elements_needed == 0:
                    break
                if len(all_lists[i]) > 0:
                    remaining_elements.append(all_lists[i].pop(0))
                    remaining_elements_needed -= 1
        # Combine the sets of elements
        return first_elements + second_elements, remaining_elements

    x_one, x_zero = return_split(
        correct_confound,
        wrong_confound,
        correct_not_confound,
        wrong_not_confound,
        correct_lab,
        wrong_lab,
        wrong_lab_img_confound,
        correct_lab_img_confound,
        corr_no_conf,
        wrong_no_conf,
        0,
        1,
        300,
        [0, 2, 4, 7, 8]
    )

    compute_auc_by_metrics(
        x_one,
        x_zero,
        "X confound",
        arguments_folder
    )

    y_one, y_zero = return_split(
        correct_confound,
        wrong_confound,
        correct_not_confound,
        wrong_not_confound,
        correct_lab,
        wrong_lab,
        wrong_lab_img_confound,
        correct_lab_img_confound,
        corr_no_conf,
        wrong_no_conf,
        4,
        5,
        300,
        [0, 2, 4, 7, 8]
    )

    compute_auc_by_metrics(
        y_one,
        y_zero,
        "Y confound",
        arguments_folder
    )

    xy_one, xy_zero = return_split(
        correct_confound,
        wrong_confound,
        correct_not_confound,
        wrong_not_confound,
        correct_lab,
        wrong_lab,
        wrong_lab_img_confound,
        correct_lab_img_confound,
        corr_no_conf,
        wrong_no_conf,
        7,
        6,
        300,
        [0, 2, 4, 7, 8]
    )

    compute_auc_by_metrics(
        xy_one,
        xy_zero,
        "XY confound",
        arguments_folder
    )


def compute_auc_by_metrics(
    one_list: List[ArgumentBucket],
    zero_list: List[ArgumentBucket],
    name: str,
    arguments_folder: str,
):
    from sklearn.metrics import roc_auc_score

    def rescale_values(value):
        return (value - np.min(value)) / (np.max(value) - np.min(value))

    def max_select(list_1, list_2):
        new_list = []
        for x, y in zip(list_1, list_2):
            new_list.append(max(x, y))
        return new_list

    def min_select(list_1, list_2):
        new_list = []
        for x, y in zip(list_1, list_2):
            new_list.append(min(x, y))
        return new_list

    def geom_mean(list_1, list_2):
        import math
        new_list = []
        for a, b in zip(list_1, list_2):
            new_list.append(math.sqrt(a * b))
        return new_list

    def sum_values(list_1, list_2):
        new_list = []
        for a, b in zip(list_1, list_2):
            new_list.append(a + b)
        return new_list

    def mul_values(list_1, list_2):
        new_list = []
        for a, b in zip(list_1, list_2):
            new_list.append(a * b)
        return new_list

    def harmonic_mean(list_1, list_2):
        new_list = []
        for a, b in zip(list_1, list_2):
            new_list.append(2 / ((1 / a) + (1 / b)))
        return new_list

    def threshold_based_method(ig_entropy_list, label_entropy_list):
        def tbased(ig_entropy, label_entropy):
            if label_entropy < 0.05:
                return 1
            elif label_entropy > 0.7:
                return 1
            else:
                return ig_entropy < 0.310
        new_list = []
        for ig_entropy, label_entropy in zip(ig_entropy_list, label_entropy_list):
            new_list.append(tbased(ig_entropy, label_entropy))
        return new_list


    print("I am processing", name)
    # Create labels: 1 for the first list, 0 for the rest
    print(len(one_list), len(zero_list))
    #  labels = np.concatenate([np.ones(len(one_list)), np.zeros((len(zero_list)))])
    labels = np.concatenate([np.zeros(len(one_list)), np.ones((len(zero_list)))])

    # Apply the custom function to objects within the lists
    ig_gradient_list = np.array(
        [item.get_maximum_ig_score() for item in one_list] +
        [item.get_maximum_ig_score() for item in zero_list]
    )
    label_gradient_list = np.array(
        [item.get_maximum_label_score() for item in one_list] +
        [item.get_maximum_label_score() for item in zero_list]
    )
    ig_entropy_list = np.array(
        [item.get_ig_entropy() for item in one_list] +
        [item.get_ig_entropy() for item in zero_list]
    )
    label_entropy_list = np.array(
        [item.get_predicted_label_entropy() for item in one_list] +
        [item.get_predicted_label_entropy() for item in zero_list]
    )
    random_value_list = np.array(
        [item.get_random_ranking() for item in one_list] +
        [item.get_random_ranking() for item in zero_list]
    )
    # Combine using maximum
    combined_entropy_max_list = np.array(
        max_select(rescale_values([item.get_ig_entropy() for item in one_list]), rescale_values([item.get_predicted_label_entropy() for item in one_list])) +
        max_select(rescale_values([item.get_ig_entropy() for item in zero_list]), rescale_values([item.get_predicted_label_entropy() for item in zero_list]))
    )
    # Combine using minimum
    combined_entropy_min_list = np.array(
        min_select(rescale_values([item.get_ig_entropy() for item in one_list]), rescale_values([item.get_predicted_label_entropy() for item in one_list])) +
        min_select(rescale_values([item.get_ig_entropy() for item in zero_list]), rescale_values([item.get_predicted_label_entropy() for item in zero_list]))
    )
    # Combine using product
    combined_entropy_prod_list = np.array(
        mul_values(rescale_values([item.get_ig_entropy() for item in one_list]), rescale_values([item.get_predicted_label_entropy() for item in one_list])) +
        mul_values(rescale_values([item.get_ig_entropy() for item in zero_list]), rescale_values([item.get_predicted_label_entropy() for item in zero_list]))
    )
    # Combine using sum
    combined_entropy_sum_list = np.array(
        sum_values(rescale_values([item.get_ig_entropy() for item in one_list]), rescale_values([item.get_predicted_label_entropy() for item in one_list])) +
        sum_values(rescale_values([item.get_ig_entropy() for item in zero_list]), rescale_values([item.get_predicted_label_entropy() for item in zero_list]))
    )
    # Combine using geometric mean
    combined_entropy_geom_list = np.array(
        geom_mean(rescale_values([item.get_ig_entropy() for item in one_list]), rescale_values([item.get_predicted_label_entropy() for item in one_list])) +
        geom_mean(rescale_values([item.get_ig_entropy() for item in zero_list]), rescale_values([item.get_predicted_label_entropy() for item in zero_list]))
    )
    # Combine using harmonic mean
    combined_entropy_harmonic_mean_list = np.array(
        harmonic_mean(rescale_values([item.get_ig_entropy() for item in one_list]), rescale_values([item.get_predicted_label_entropy() for item in one_list])) +
        harmonic_mean(rescale_values([item.get_ig_entropy() for item in zero_list]), rescale_values([item.get_predicted_label_entropy() for item in zero_list]))
    )
    # Combine
    combine_custom_3_threshold = np.array(
        threshold_based_method([item.get_ig_entropy() for item in one_list], [item.get_predicted_label_entropy() for item in one_list]) +
        threshold_based_method([item.get_ig_entropy() for item in zero_list], [item.get_predicted_label_entropy() for item in zero_list])
    )

    # lists and names
    lists = [ig_gradient_list, label_gradient_list, ig_entropy_list, label_entropy_list, random_value_list, combined_entropy_max_list, combined_entropy_min_list, combined_entropy_prod_list, combined_entropy_sum_list, combined_entropy_geom_list, combined_entropy_harmonic_mean_list, combine_custom_3_threshold]
    names = ["IG gradient", "Label gradient", "IG entropy", "Label entropy", "Random", "Max", "Min", "Prod", "Sum", "Geom", "Harmonic", "Thresholds"]
    print(labels)
    for n, l in zip(names, lists):
        print(n, l)
        print(roc_auc_score(labels, l))
    aucs = [roc_auc_score(labels, l) for l in lists]

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create the table
    table_data = [[name, f'{value:.2f}'] for name, value in zip(names, aucs)]
    table = ax.table(cellText=table_data, colLabels=["Approach", "AUC"], cellLoc='center', loc='center')

    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    # Center-align text in cells
    for key, cell in table._cells.items():
        cell.set_text_props(ha="center", va="center")

    # Color the header cells
    header_cells = table._cells[(0, 0)], table._cells[(0, 1)]
    for cell in header_cells:
        cell.set_facecolor("#40466e")
        cell.set_text_props(color="white")

    # Color alternate rows and add a subtle background color
    for i, row_idx in enumerate(range(1, len(table_data) + 1)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        row_cells = [table._cells[(row_idx, 0)], table._cells[(row_idx, 1)]]
        for cell in row_cells:
            cell.set_facecolor(color)

    # Hide axis
    ax.axis("off")

    # Add a title to the plot
    title = ax.set_title("{} Approaches".format(name), fontsize=16, weight="bold")

    # Adjust title position
    title.set_position([.5, 1.25])

    # Show the plot
    plt.subplots_adjust(top=0.8)  # Adjust top margin for title

    # save the figure
    plt.savefig("{}/{}_plot.pdf".format(arguments_folder, name))
    plt.show()

def set_common_color_palette():
    return sns.color_palette('Set2')

def save_and_close_plot(arguments_folder, plot_name):
    plt.savefig(os.path.join(arguments_folder, plot_name))
    plt.close()

def plot_violin(data_sets, labels, title, x_label, y_label, arguments_folder, plot_name):
    colors = set_common_color_palette()
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data_sets, showmedians=True, palette=colors)
    plt.xticks(range(len(data_sets)), labels)
    plt.xlabel(x_label)
    plt.title(title, fontsize=14)
    plt.ylabel(y_label, fontsize=12)
    save_and_close_plot(arguments_folder, plot_name)

def plot_box(data_sets, labels, title, x_label, y_label, arguments_folder, plot_name):
    colors = set_common_color_palette()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_sets, palette=colors)
    plt.xticks(range(len(data_sets)), labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    save_and_close_plot(arguments_folder, plot_name)

def plot_histogram(data_sets, labels, title, x_label, y_label, arguments_folder, plot_name):
    colors = set_common_color_palette()
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data_sets, bins=20, alpha=0.7, color=colors[:len(data_sets)], label=labels)
    jitter = 0
    for containers in patches:
        for patch in containers:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    save_and_close_plot(arguments_folder, plot_name)

def plot_kde(data_sets, labels, title, x_label, arguments_folder, plot_name):
    colors = set_common_color_palette()
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(data_sets):
        sns.kdeplot(data, color=colors[i], label=labels[i])
    plt.legend()
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    save_and_close_plot(arguments_folder, plot_name)

def plot_swarm(data_sets, labels, title, x_label, y_label, arguments_folder, plot_name):
    colors = set_common_color_palette()
    plt.figure(figsize=(10, 6))
    concatenated_data = []
    for i, data in enumerate(data_sets):
        concatenated_data.extend([i] * len(data))
    sns.swarmplot(x=concatenated_data, y=np.concatenate(data_sets), palette=colors[:len(data_sets)])
    plt.xticks(range(len(data_sets)), labels)
    plt.title(title, fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label, fontsize=12)
    save_and_close_plot(arguments_folder, plot_name)

def plot_cdf(data_sets, labels, title, x_label, arguments_folder, plot_name):
    colors = set_common_color_palette()
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(data_sets):
        plt.hist(data, density=True, cumulative=True, histtype='step', label=labels[i], bins=100, linewidth=2, color=colors[i])
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)
    save_and_close_plot(arguments_folder, plot_name)


def create_combined_plots(data_sets, labels, titles, x_labels, y_labels, arguments_folder, plot_name, plot_positions=None, figsize=(16, 12)):
    colors = set_common_color_palette()
    num_plots = len(data_sets)

    # Create a new figure for the combined plots
    plt.figure(figsize=figsize)
    plt.suptitle(f'Combined Plots [{plot_name}]', fontsize=16)

    if plot_positions is None:
        plot_positions = [i+231 for i in range(num_plots)]

    for i in range(num_plots):
        plt.subplot(plot_positions[i])
        if isinstance(data_sets[i], list):
            data = data_sets[i]
        else:
            data = [data_sets[i]]
        if isinstance(labels[i], list):
            label = labels[i]
        else:
            label = labels[i]
        sns.violinplot(data=data, showmedians=True, palette=colors)
        plt.xticks([0], label)  # Use [0] to set a single tick position for the violin plot
        plt.xlabel(x_labels[i])
        plt.title(titles[i], fontsize=14)
        plt.ylabel(y_labels[i], fontsize=12)

    # Save the combined plot
    save_and_close_plot(arguments_folder, f'combined_plots_{plot_name}.pdf')


def plot_arguments(
    correct_confound: ArgumentsStepArgs,
    wrong_confound: ArgumentsStepArgs,
    correct_not_confound: ArgumentsStepArgs,
    wrong_not_confound: ArgumentsStepArgs,
    correct_lab: ArgumentsStepArgs,
    wrong_lab: ArgumentsStepArgs,
    wrong_lab_img_confound: ArgumentsStepArgs,
    correct_lab_img_confound: ArgumentsStepArgs,
    corr_no_conf: ArgumentsStepArgs,
    wrong_no_conf: ArgumentsStepArgs,
    arguments_folder: str,
    tau: float,
) -> None:
    """Produce all the plots for the arguments
    Args:
        correct_confound [ArgumentsStepArgs]: corresct and confound statistics
        wrong_confound [ArgumentsStepArgs]: wrong and confound statistics
        correct_not_confound [ArgumentsStepArgs]: correct not confound statistics
        wrong_not_confound [ArgumentsStepArgs]: wrong not confound statistics
        correct_lab [ArgumentsStepArgs]: correct label statistics
        wrong_lab [ArgumentsStepArgs]: wrong label statistics
        wrong_lab_img [ArgumentsStepArgs]: wrongly predicted samples both images and label confoudners statistics
        arguments_folder [str]: arguments folder
    """

    # MAX start
    conf_list_image = list(
        itertools.chain(
            [
                True
                for _ in range(len(correct_confound.max_ig_label_list_for_score_plot))
            ],
            [True for _ in range(len(wrong_confound.max_ig_label_list_for_score_plot))],
            [
                False
                for _ in range(
                    len(correct_not_confound.max_ig_label_list_for_score_plot)
                )
            ],
            [
                False
                for _ in range(len(wrong_not_confound.max_ig_label_list_for_score_plot))
            ],
        )
    )
    correct_list_image = list(
        itertools.chain(
            [
                True
                for _ in range(len(correct_confound.max_ig_label_list_for_score_plot))
            ],
            [
                False
                for _ in range(len(wrong_confound.max_ig_label_list_for_score_plot))
            ],
            [
                True
                for _ in range(
                    len(correct_not_confound.max_ig_label_list_for_score_plot)
                )
            ],
            [
                False
                for _ in range(len(wrong_not_confound.max_ig_label_list_for_score_plot))
            ],
        )
    )
    ig_list_image = list(
        itertools.chain(
            correct_confound.max_ig_label_list_for_score_plot,
            wrong_confound.max_ig_label_list_for_score_plot,
            correct_not_confound.max_ig_label_list_for_score_plot,
            wrong_not_confound.max_ig_label_list_for_score_plot,
        )
    )
    ig_list_label = list(
        itertools.chain(
            correct_lab.max_ig_label_list_for_score_plot,
            wrong_lab.max_ig_label_list_for_score_plot,
        )
    )
    correct_list_label = list(
        itertools.chain(
            [True for _ in range(len(correct_lab.max_ig_label_list_for_score_plot))],
            [False for _ in range(len(wrong_lab.max_ig_label_list_for_score_plot))],
        )
    )
    ig_list_label_and_image = list(
        itertools.chain(
            correct_lab_img_confound.max_ig_label_list_for_score_plot,
            wrong_lab_img_confound.max_ig_label_list_for_score_plot,
        )
    )
    correct_list_label_and_image = list(
        itertools.chain(
            [
                True
                for _ in range(
                    len(correct_lab_img_confound.max_ig_label_list_for_score_plot)
                )
            ],
            [
                False
                for _ in range(
                    len(wrong_lab_img_confound.max_ig_label_list_for_score_plot)
                )
            ],
        )
    )

    label_entropy_conf = np.array(
        [float(item) for item in correct_lab.label_entropy_list] +
        [float(item) for item in wrong_lab.label_entropy_list]
    )

    label_entropy_not_conf = np.array(
        [float(item) for item in corr_no_conf.label_entropy_list] +
        [float(item) for item in wrong_no_conf.label_entropy_list]
    )

    organize_aucs(
        correct_confound,
        wrong_confound,
        correct_not_confound,
        wrong_not_confound,
        correct_lab,
        wrong_lab,
        wrong_lab_img_confound,
        correct_lab_img_confound,
        corr_no_conf,
        wrong_no_conf,
        arguments_folder,
    )

    data_sets = [label_entropy_conf, label_entropy_not_conf]
    labels = ['With Sample Imbalance', 'Without Sample Imbalance']

    plot_violin(data_sets, labels, 'Violin Plot [Label Entropy]', '', 'Label Entropy', arguments_folder, 'violin_plot_label_entropy.pdf')
    plot_box(data_sets, labels, 'Box Plot [Label Entropy]', '', 'Label Entropy', arguments_folder, 'box_plot_label_entropy.pdf')
    plot_histogram(data_sets, labels, 'Histogram [Label Entropy]', 'Label Entropy', 'Label Entropy', arguments_folder, 'histogram_label_entropy.pdf')
    plot_kde(data_sets, labels, 'KDE Plot [Label Entropy]', 'Label Entropy', arguments_folder, 'kde_plot_label_entropy.pdf')
    plot_swarm(data_sets, labels, 'Swarm Plot [Label Entropy]', '', 'Label Entropy', arguments_folder, 'swarm_plot_label_entropy.pdf')
    plot_cdf(data_sets, labels, 'CDF Plot [Label Entropy]', 'Label Entropy', arguments_folder, 'cdf_plot_label_entropy.pdf')

    labels = [['With Sample Imbalance'], ['Without Sample Imbalance']]
    titles = ['Violin Plot', 'Box Plot']
    x_labels = ['', '']
    y_labels = ['Label Entropy', 'Label Entropy']
    create_combined_plots(data_sets, labels, titles, x_labels, y_labels, arguments_folder, 'label_entropy', figsize=(16, 12))

    exit(0)

    # Combined Plots
    plt.suptitle('Combined Plots [Label Entropy]', fontsize=16)

    # Violin Plot
    plt.subplot(231)
    sns.violinplot(data=[label_entropy_conf, label_entropy_not_conf],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    # Box Plot
    plt.subplot(232)
    sns.boxplot(data=[label_entropy_conf, label_entropy_not_conf],
            palette=colors)
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    # Histogram
    plt.subplot(233)
    plt.hist([label_entropy_conf, label_entropy_not_conf], bins=20, alpha=0.7, color=colors[:2], label=['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    # KDE Plot
    plt.subplot(234)
    sns.kdeplot(label_entropy_conf, color=colors[0], label='With Sample Imbalance')
    sns.kdeplot(label_entropy_not_conf, color=colors[1], label='Without Sample Imbalance')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    # Swarm Plot
    plt.subplot(235)
    sns.swarmplot(x=np.concatenate([np.zeros_like(label_entropy_conf), np.ones_like(label_entropy_not_conf)]),
                  y=np.concatenate([label_entropy_conf, label_entropy_not_conf]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.title('Swarm Plot', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Label Entropy', fontsize=12)

    # CDF Plot
    plt.subplot(236)
    plt.hist(label_entropy_conf, density=True, cumulative=True, histtype='step', label='With Sample Imbalance', bins=100, linewidth=2, color=colors[0])
    plt.hist(label_entropy_not_conf, density=True, cumulative=True, histtype='step', label='Without Sample Imbalance', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save the combined plot
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_label_entropy.pdf'))
    plt.close()

    # NOW THE DIFFERENCES BETWEEN CORRECTAND NOT CORRECT!
    lab_conf_correct=np.array([float(item) for item in correct_lab.label_entropy_list])
    lab_conf_wrong=np.array([float(item) for item in wrong_lab.label_entropy_list])
    not_lab_conf_correct = np.array([float(item) for item in corr_no_conf.label_entropy_list])
    not_lab_conf_wrong = np.array([float(item) for item in wrong_no_conf.label_entropy_list])
    not_lab_conf_but_watermar_correct = np.array([float(item) for item in correct_confound.label_entropy_list])
    not_lab_conf_but_watermar_wrong = np.array([float(item) for item in wrong_confound.label_entropy_list])

    data = {
        'Label Entropy': np.concatenate([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong]),
        'Prediction': ['With Sample Imbalance (Correct)'] * len(lab_conf_correct) + ['With Sample Imbalance (Wrong)'] *  len(lab_conf_wrong) + ['Without Sample Imbalance (Correct)'] * len(not_lab_conf_correct) + ['Without Sample Imbalance (Wrong)'] * len(not_lab_conf_wrong)
    }

    # Create DataFrame
    data_df = pd.DataFrame(data)

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=6)

    # With all
    # Violin Plot
    plt.figure(figsize=(10, 12))
    sns.violinplot(data=[lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong, not_lab_conf_but_watermar_correct, not_lab_conf_but_watermar_wrong], showmedians=True, palette=colors[:6])
    plt.xticks([0, 1, 2, 3, 4, 5], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)', 'With Watermark (Correct)', 'With Watermark (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Violin Plot [Label Entropy]', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'violin_label_entropy_by_color_all.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 12))
    sns.boxplot(data=[lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong, not_lab_conf_but_watermar_correct, not_lab_conf_but_watermar_wrong], palette=colors[:6])
    plt.xticks([0, 1, 2, 3, 4, 5], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)', 'With Watermark (Correct)', 'With Watermark (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Box Plot [Label Entropy]', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'boxplot_label_entropy_by_color_all.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong, not_lab_conf_but_watermar_correct, not_lab_conf_but_watermar_wrong],
             bins=20, alpha=0.7, color=colors[:6], label=['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)', 'With Watermark (Correct)', 'With Watermark (Wrong)'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)


    plt.legend()
    plt.title('Histogram [Label Entropy]', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'hist_label_entropy_by_color_all.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(lab_conf_correct, color=colors[0], label='With Sample Imbalance (Correct)')
    sns.kdeplot(lab_conf_wrong, color=colors[1], label='With Sample Imbalance (Wrong)')
    sns.kdeplot(not_lab_conf_correct, color=colors[2], label='Without Sample Imbalance (Correct)')
    sns.kdeplot(not_lab_conf_wrong, color=colors[3], label='Without Sample Imbalance (Wrong)')
    sns.kdeplot(not_lab_conf_but_watermar_correct, color=colors[4], label='With Watermark (Correct)')
    sns.kdeplot(not_lab_conf_but_watermar_wrong, color=colors[5], label='With Watermark (Wrong)')
    plt.legend()
    plt.title('KDE Plot [Label Entropy]', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'kde_label_entropy_by_color_all.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 12))
    sns.swarmplot(x=np.concatenate([np.zeros_like(lab_conf_correct), np.ones_like(lab_conf_wrong),
                                    2*np.ones_like(not_lab_conf_correct), 3*np.ones_like(not_lab_conf_wrong), 4*np.ones_like(not_lab_conf_but_watermar_correct), 5*np.ones_like(not_lab_conf_but_watermar_wrong)]),
                  y=np.concatenate([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong, not_lab_conf_but_watermar_correct, not_lab_conf_but_watermar_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3, 4, 5], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)', 'With Watermark (Correct)', 'With Watermark (Wrong)'], rotation=21)
    plt.xlabel('')
    plt.title('Swarm Plot [Label Entropy]', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'swarm_label_entropy_by_color_all.pdf'))
    plt.close()

    # NOT WITH ALL
    # Violin Plot
    plt.figure(figsize=(10, 12))
    sns.violinplot(data=[lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong], showmedians=True, palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Violin Plot [Label Entropy]', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)
    plt.savefig(os.path.join(arguments_folder, 'violin_label_entropy_by_color.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 12))
    sns.boxplot(data=[lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong], palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Box Plot [Label Entropy]', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'boxplot_label_entropy_by_color.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong],
             bins=20, alpha=0.7, color=colors[:4], label=['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)


    plt.legend()
    plt.title('Histogram [Label Entropy]', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'hist_label_entropy_by_color.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(lab_conf_correct, color=colors[0], label='With Sample Imbalance (Correct)')
    sns.kdeplot(lab_conf_wrong, color=colors[1], label='With Sample Imbalance (Wrong)')
    sns.kdeplot(not_lab_conf_correct, color=colors[2], label='Without Sample Imbalance (Correct)')
    sns.kdeplot(not_lab_conf_wrong, color=colors[3], label='Without Sample Imbalance (Wrong)')
    plt.legend()
    plt.title('KDE Plot [Label Entropy]', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'kde_label_entropy_by_color.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 12))
    sns.swarmplot(x=np.concatenate([np.zeros_like(lab_conf_correct), np.ones_like(lab_conf_wrong),
                                    2*np.ones_like(not_lab_conf_correct), 3*np.ones_like(not_lab_conf_wrong)]),
                  y=np.concatenate([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'], rotation=21)
    plt.xlabel('')
    plt.title('Swarm Plot [Label Entropy]', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'swarm_label_entropy_by_color.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 12))
    sns.histplot(data_df, x='Label Entropy', cumulative=True, hue='Prediction', element='step', palette=colors)
    plt.xlabel('Label Entropy', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Input Gradient CDF Plot with Hue [Label Entropy]', fontsize=14)
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='With Sample Imbalance (Correct)'),
                   plt.Line2D([0], [0], color=colors[1], lw=2, label='With Sample Imbalance (Wrong)'),
                   plt.Line2D([0], [0], color=colors[2], lw=2, label='Without Sample Imbalance (Correct)'),
                   plt.Line2D([0], [0], color=colors[3], lw=2, label='Without Sample Imbalance (Wrong)')]

    # Add the legend to the plot
    plt.legend(handles=legend_elements, title='Prediction', title_fontsize=12, loc='lower right')

    plt.savefig(os.path.join(arguments_folder, 'cdf_by_color.pdf'))
    plt.close()

    # Combined
    plt.figure(figsize=(12, 12))

    plt.suptitle('Combined Plots [Label Entropy]', fontsize=16)

    plt.subplot(321)
    sns.violinplot(data=[lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong], showmedians=True, palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'], rotation=21)
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.subplot(322)
    sns.boxplot(data=[lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong], palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'], rotation=21)
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    plt.subplot(323)
    plt.hist([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong],
             bins=20, alpha=0.7, color=colors[:4], label=['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    plt.subplot(324)
    sns.kdeplot(lab_conf_correct, color=colors[0], label='With Sample Imbalance (Correct)')
    sns.kdeplot(lab_conf_wrong, color=colors[1], label='With Sample Imbalance (Wrong)')
    sns.kdeplot(not_lab_conf_correct, color=colors[2], label='Without Sample Imbalance (Correct)')
    sns.kdeplot(not_lab_conf_wrong, color=colors[3], label='Without Sample Imbalance (Wrong)')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Label Entropy', fontsize=12)

    plt.subplot(325)
    sns.swarmplot(x=np.concatenate([np.zeros_like(lab_conf_correct), np.ones_like(lab_conf_wrong),
                                    2*np.ones_like(not_lab_conf_correct), 3*np.ones_like(not_lab_conf_wrong)]),
                  y=np.concatenate([lab_conf_correct, lab_conf_wrong, not_lab_conf_correct, not_lab_conf_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Swarm Plot', fontsize=14)
    plt.ylabel('Label Entropy', fontsize=12)

    plt.subplot(326)
    sns.histplot(data_df, x='Label Entropy', cumulative=True, hue='Prediction', element='step', palette=colors)
    plt.xlabel('Label Entropy', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Input Gradient CDF Plot with Hue', fontsize=14)
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='With Sample Imbalance (Correct)'),
                   plt.Line2D([0], [0], color=colors[1], lw=2, label='With Sample Imbalance (Wrong)'),
                   plt.Line2D([0], [0], color=colors[2], lw=2, label='Without Sample Imbalance (Correct)'),
                   plt.Line2D([0], [0], color=colors[3], lw=2, label='Without Sample Imbalance (Wrong)')]
    # Add the legend to the plot
    plt.legend(handles=legend_elements, title='Prediction', title_fontsize=12, loc='lower right')

    # Add space between subplots
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_label_entropy_by_color.pdf'))
    plt.close()

    # IG ENTROPY

    input_gradients_entropy_with_watermarks = np.array(
        [float(item) for item in correct_confound.ig_entropy_list] +
        [float(item) for item in wrong_confound.ig_entropy_list]
    )

    input_gradients_entropy_without_watermarks = np.array(
        [float(item) for item in correct_not_confound.ig_entropy_list] +
        [float(item) for item in wrong_not_confound.ig_entropy_list]
    )

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Violin Plot [Input Gradient Entropy]', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    # Save Violin Plot
    plt.savefig(os.path.join(arguments_folder, 'violin_plot_ig_entropy.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks],
            palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.ylabel('Input Gradient Entropy', fontsize=12)
    plt.title('Box Plot [Input Gradient Entropy]', fontsize=14)

    # Save Box Plot
    plt.savefig(os.path.join(arguments_folder, 'box_plot_ig_entropy.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks],
             bins=20, alpha=0.7, color=colors[:2], label=['With Watermarks', 'Without Watermarks'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.title('Histogram [Input Gradient Entropy]', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    # Save Histogram
    plt.savefig(os.path.join(arguments_folder, 'histogram_ig_entropy.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(input_gradients_entropy_with_watermarks, color=colors[0], label='With Watermarks')
    sns.kdeplot(input_gradients_entropy_without_watermarks, color=colors[1], label='Without Watermarks')
    plt.legend()
    plt.title('KDE Plot [Input Gradient Entropy]', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    # Save KDE Plot
    plt.savefig(os.path.join(arguments_folder, 'kde_plot_ig_entropy.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x=np.concatenate([np.zeros_like(input_gradients_entropy_with_watermarks), np.ones_like(input_gradients_entropy_without_watermarks)]),
                  y=np.concatenate([input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.title('Swarm Plot [Input Gradient Entropy]', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    # Save Swarm Plot
    plt.savefig(os.path.join(arguments_folder, 'swarm_plot_ig_entropy.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 6))
    plt.hist(input_gradients_entropy_with_watermarks, density=True, cumulative=True, histtype='step', label='With Watermarks', bins=100, linewidth=2, color=colors[0])
    plt.hist(input_gradients_entropy_without_watermarks, density=True, cumulative=True, histtype='step', label='Without Watermarks', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot [Input Gradient Entropy]', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save CDF Plot
    plt.savefig(os.path.join(arguments_folder, 'cdf_plot_ig_entropy.pdf'))
    plt.close()

    # Create a new figure for the combined plots
    plt.figure(figsize=(16, 12))

    # Combined Plots
    plt.suptitle('Combined Plots [Input Gradient Entropy]', fontsize=16)

    # Violin Plot
    plt.subplot(231)
    sns.violinplot(data=[input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    # Box Plot
    plt.subplot(232)
    sns.boxplot(data=[input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks],
            palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    # Histogram
    plt.subplot(233)
    plt.hist([input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks], bins=20, alpha=0.7, color=colors[:2], label=['With Watermarks', 'Without Watermarks'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    # KDE Plot
    plt.subplot(234)
    sns.kdeplot(input_gradients_entropy_with_watermarks, color=colors[0], label='With Watermarks')
    sns.kdeplot(input_gradients_entropy_without_watermarks, color=colors[1], label='Without Watermarks')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    # Swarm Plot
    plt.subplot(235)
    sns.swarmplot(x=np.concatenate([np.zeros_like(input_gradients_entropy_with_watermarks), np.ones_like(input_gradients_entropy_without_watermarks)]),
                  y=np.concatenate([input_gradients_entropy_with_watermarks, input_gradients_entropy_without_watermarks]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.title('Swarm Plot', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    # CDF Plot
    plt.subplot(236)
    plt.hist(input_gradients_entropy_with_watermarks, density=True, cumulative=True, histtype='step', label='With Watermarks', bins=100, linewidth=2, color=colors[0])
    plt.hist(input_gradients_entropy_without_watermarks, density=True, cumulative=True, histtype='step', label='Without Watermarks', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save the combined plot
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_ig_entropy.pdf'))
    plt.close()

    # NOW THE DIFFERENCES BETWEEN GRADIENTS!
    ig_entropy_watermarked_correct=np.array([float(item) for item in correct_confound.ig_entropy_list])
    ig_entropy_watermarked_wrong=np.array([float(item) for item in wrong_confound.ig_entropy_list])
    ig_entropy_not_watermarked_correct = np.array([float(item) for item in correct_not_confound.ig_entropy_list])
    ig_entropy_not_watermarked_wrong = np.array([float(item) for item in wrong_not_confound.ig_entropy_list])

    data = {
        'Input Gradient Entropy': np.concatenate([ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong]),
        'Prediction': ['With Watermarks (Correct)'] * len(ig_entropy_watermarked_correct) + ['With Watermarks (Wrong)'] *  len(ig_entropy_watermarked_wrong) + ['Without Watermarks (Correct)'] * len(ig_entropy_not_watermarked_correct) + ['Without Watermarks (Wrong)'] * len(ig_entropy_not_watermarked_wrong)
    }

    # Create DataFrame
    data_df = pd.DataFrame(data)

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=4)

    # Violin Plot
    plt.figure(figsize=(10, 12))
    sns.violinplot(data=[ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong], showmedians=True, palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Violin Plot [Input Gradient Entropy]', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'violin_by_color_ig_entropy.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 12))
    sns.boxplot(data=[ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong], palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Box Plot [Input Gradient Entropy]', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'boxplot_by_color_ig_entropy.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong],
             bins=20, alpha=0.7, color=colors[:4], label=['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)


    plt.legend()
    plt.title('Histogram [Input Gradient Entropy]', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'hist_by_color_ig_entropy.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(ig_entropy_watermarked_correct, color=colors[0], label='With Watermarks (Correct)')
    sns.kdeplot(ig_entropy_watermarked_wrong, color=colors[1], label='With Watermarks (Wrong)')
    sns.kdeplot(ig_entropy_not_watermarked_correct, color=colors[2], label='Without Watermarks (Correct)')
    sns.kdeplot(ig_entropy_not_watermarked_wrong, color=colors[3], label='Without Watermarks (Wrong)')
    plt.legend()
    plt.title('KDE Plot [Input Gradient Entropy]', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'kde_by_color_ig_entropy.pdf'))
    plt.close()

    # Swarm Plot

    plt.figure(figsize=(10, 12))
    sns.swarmplot(x=np.concatenate([np.zeros_like(ig_entropy_watermarked_correct), np.ones_like(ig_entropy_watermarked_wrong),
                                    2*np.ones_like(ig_entropy_not_watermarked_correct), 3*np.ones_like(ig_entropy_not_watermarked_wrong)]),
                  y=np.concatenate([ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3], ['Watermarked (Correct)', 'Watermarked (Wrong)', 'Not Watermarked (Correct)', 'Not Watermarked (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Swarm Plot [Input Gradient Entropy]', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'swarm_by_color_ig_entropy.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 12))
    sns.histplot(data_df, x='Input Gradient Entropy', cumulative=True, hue='Prediction', element='step', palette=colors)
    plt.xlabel('Input Gradient Entropy', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Input Gradient CDF Plot with Hue [Input Gradient Entropy]', fontsize=14)
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='With Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[1], lw=2, label='With Watermarks (Wrong)'),
                   plt.Line2D([0], [0], color=colors[2], lw=2, label='Without Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[3], lw=2, label='Without Watermarks (Wrong)')]

    # Add the legend to the plot
    plt.legend(handles=legend_elements, title='Prediction', title_fontsize=12, loc='lower right')

    plt.savefig(os.path.join(arguments_folder, 'cdf_by_color_ig_entropy.pdf'))
    plt.close()

    # Combined
    plt.figure(figsize=(12, 12))

    plt.suptitle('Combined Plots [Input Gradient Entropy]', fontsize=16)

    plt.subplot(321)
    sns.violinplot(data=[ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong], showmedians=True, palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    plt.subplot(322)
    sns.boxplot(data=[ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong], palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    plt.subplot(323)
    plt.hist([ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong],
             bins=20, alpha=0.7, color=colors[:4], label=['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    plt.subplot(324)
    sns.kdeplot(ig_entropy_watermarked_correct, color=colors[0], label='With Watermarks (Correct)')
    sns.kdeplot(ig_entropy_watermarked_wrong, color=colors[1], label='With Watermarks (Wrong)')
    sns.kdeplot(ig_entropy_not_watermarked_correct, color=colors[2], label='Without Watermarks (Correct)')
    sns.kdeplot(ig_entropy_not_watermarked_wrong, color=colors[3], label='Without Watermarks (Wrong)')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Input Gradient Entropy', fontsize=12)

    plt.subplot(325)
    sns.swarmplot(x=np.concatenate([np.zeros_like(ig_entropy_watermarked_correct), np.ones_like(ig_entropy_watermarked_wrong),
                                    2*np.ones_like(ig_entropy_not_watermarked_correct), 3*np.ones_like(ig_entropy_not_watermarked_wrong)]),
                  y=np.concatenate([ig_entropy_watermarked_correct, ig_entropy_watermarked_wrong, ig_entropy_not_watermarked_correct, ig_entropy_not_watermarked_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3], ['Watermarked (Correct)', 'Watermarked (Wrong)', 'Not Watermarked (Correct)', 'Not Watermarked (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Swarm Plot', fontsize=14)
    plt.ylabel('Input Gradient Entropy', fontsize=12)

    plt.subplot(326)
    sns.histplot(data_df, x='Input Gradient Entropy', cumulative=True, hue='Prediction', element='step', palette=colors)
    plt.xlabel('Input Gradient Entropy', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Input Gradient CDF Plot with Hue', fontsize=14)
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='With Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[1], lw=2, label='With Watermarks (Wrong)'),
                   plt.Line2D([0], [0], color=colors[2], lw=2, label='Without Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[3], lw=2, label='Without Watermarks (Wrong)')]
    # Add the legend to the plot
    plt.legend(handles=legend_elements, title='Prediction', title_fontsize=12, loc='lower right')

    # Add space between subplots
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_by_color_ig_entropy.pdf'))
    plt.close()

    print("Evviva", len(correct_confound.max_ig_label_list_for_score_plot), type(correct_confound.max_ig_label_list_for_score_plot))

    print(correct_confound.max_ig_label_list_for_score_plot)

    input_gradients_with_watermarks = np.array(
        [float(item[0][0]) for item in correct_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_confound.max_ig_label_list_for_score_plot]
    )
    input_gradients_without_watermarks = np.array(
        [float(item[0][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot]
    )

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2')

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[input_gradients_with_watermarks, input_gradients_without_watermarks],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Violin Plot [Input Gradients]', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    # Save Violin Plot
    plt.savefig(os.path.join(arguments_folder, 'violin_plot.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[input_gradients_with_watermarks, input_gradients_without_watermarks],
            palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.ylabel('Input Gradient Magnitude', fontsize=12)
    plt.title('Box Plot [Input Gradients]', fontsize=14)

    # Save Box Plot
    plt.savefig(os.path.join(arguments_folder, 'box_plot.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([input_gradients_with_watermarks, input_gradients_without_watermarks],
             bins=20, alpha=0.7, color=colors[:2], label=['With Watermarks', 'Without Watermarks'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.title('Histogram [Input Gradients]', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    # Save Histogram
    plt.savefig(os.path.join(arguments_folder, 'histogram.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(input_gradients_with_watermarks, color=colors[0], label='With Watermarks')
    sns.kdeplot(input_gradients_without_watermarks, color=colors[1], label='Without Watermarks')
    plt.legend()
    plt.title('KDE Plot [Input Gradients]', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    # Save KDE Plot
    plt.savefig(os.path.join(arguments_folder, 'kde_plot.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x=np.concatenate([np.zeros_like(input_gradients_with_watermarks), np.ones_like(input_gradients_without_watermarks)]),
                  y=np.concatenate([input_gradients_with_watermarks, input_gradients_without_watermarks]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.title('Swarm Plot [Input Gradients]', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    # Save Swarm Plot
    plt.savefig(os.path.join(arguments_folder, 'swarm_plot.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 6))
    plt.hist(input_gradients_with_watermarks, density=True, cumulative=True, histtype='step', label='With Watermarks', bins=100, linewidth=2, color=colors[0])
    plt.hist(input_gradients_without_watermarks, density=True, cumulative=True, histtype='step', label='Without Watermarks', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot [Input Gradients]', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save CDF Plot
    plt.savefig(os.path.join(arguments_folder, 'cdf_plot.pdf'))
    plt.close()

    # Create a new figure for the combined plots
    plt.figure(figsize=(16, 12))

    # Combined Plots
    plt.suptitle('Combined Plots [Input Gradients]', fontsize=16)

    # Violin Plot
    plt.subplot(231)
    sns.violinplot(data=[input_gradients_with_watermarks, input_gradients_without_watermarks],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    # Box Plot
    plt.subplot(232)
    sns.boxplot(data=[input_gradients_with_watermarks, input_gradients_without_watermarks],
            palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    # Histogram
    plt.subplot(233)
    plt.hist([input_gradients_with_watermarks, input_gradients_without_watermarks], bins=20, alpha=0.7, color=colors[:2], label=['With Watermarks', 'Without Watermarks'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    # KDE Plot
    plt.subplot(234)
    sns.kdeplot(input_gradients_with_watermarks, color=colors[0], label='With Watermarks')
    sns.kdeplot(input_gradients_without_watermarks, color=colors[1], label='Without Watermarks')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    # Swarm Plot
    plt.subplot(235)
    sns.swarmplot(x=np.concatenate([np.zeros_like(input_gradients_with_watermarks), np.ones_like(input_gradients_without_watermarks)]),
                  y=np.concatenate([input_gradients_with_watermarks, input_gradients_without_watermarks]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.title('Swarm Plot', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    # CDF Plot
    plt.subplot(236)
    plt.hist(input_gradients_with_watermarks, density=True, cumulative=True, histtype='step', label='With Watermarks', bins=100, linewidth=2, color=colors[0])
    plt.hist(input_gradients_without_watermarks, density=True, cumulative=True, histtype='step', label='Without Watermarks', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save the combined plot
    plt.savefig(os.path.join(arguments_folder, 'combined_plots.pdf'))
    plt.close()

    # Correlation
    # Calculate correlation coefficients
    from scipy import stats
    size_min = min([len(input_gradients_with_watermarks), len(input_gradients_without_watermarks)])
    pearson_corr, pearson_p_value = stats.pearsonr(input_gradients_with_watermarks[:size_min], input_gradients_without_watermarks[:size_min])
    spearman_corr, spearman_p_value = stats.spearmanr(input_gradients_with_watermarks[:size_min], input_gradients_without_watermarks[:size_min])

    # Create a DataFrame to store the data
    data_df = pd.DataFrame({
        'Input Gradient Magnitude': np.concatenate([input_gradients_without_watermarks, input_gradients_with_watermarks]),
        'Watermarked': ['No'] * len(input_gradients_without_watermarks) + ['Yes'] * len(input_gradients_with_watermarks)
    })

    # Set a common color palette for both groups
    colors = sns.color_palette('Set2', n_colors=2)

    # Set a larger figure size for better visualizations
    plt.figure(figsize=(10, 6))

    # Scatter Plot
    plt.subplot(121)
    sns.scatterplot(data=data_df, x='Input Gradient Magnitude', y='Watermarked', hue='Watermarked', palette=colors, legend=False)
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Watermarked')
    plt.yticks([0, 1], ['No', 'Yes'])

    # Add subtitles for Pearson correlation and p-value
    plt.title(f"Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_p_value:.4f}")

    # Regression Plot
    plt.subplot(122)

    data_df = pd.DataFrame({
        'Input Gradient Magnitude': np.concatenate([input_gradients_with_watermarks, input_gradients_without_watermarks]),
        'Watermarked': ['With Watermarks'] * len(input_gradients_with_watermarks) + ['Without Watermarks'] * len(input_gradients_without_watermarks)
    })
    # Convert 'With Watermarks' to 1 and 'Without Watermarks' to 0
    data_df['Watermarked'] = data_df['Watermarked'].map({'With Watermarks': 1, 'Without Watermarks': 0})

    sns.regplot(data=data_df, x='Input Gradient Magnitude', y='Watermarked', logistic=True, scatter_kws={'color': colors[0]}, line_kws={'color': colors[1]})

    # Invert the y-axis representation
    plt.gca().invert_yaxis()

    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Watermarked')
    plt.yticks([0, 1], ['No', 'Yes'])
    # Add subtitles for Spearman correlation and p-value
    plt.title(f"Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p_value:.4f}")

    # Save the figure
    plt.suptitle('Scatter Plot and Logistic Regression Plot of Input Gradient Magnitude vs. Watermarked')
    plt.savefig(os.path.join(arguments_folder, 'correlation_plots.pdf'))
    plt.close()

    # NOW THE DIFFERENCES BETWEEN GRADIENTS!
    watermarked_correct=np.array([float(item[0][0]) for item in correct_confound.max_ig_label_list_for_score_plot])
    watermarked_wrong=np.array([float(item[0][0]) for item in wrong_confound.max_ig_label_list_for_score_plot])
    not_watermarked_correct = np.array([float(item[0][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot])
    not_watermarked_wrong = np.array([float(item[0][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot])

    data = {
        'Input Gradient Magnitude': np.concatenate([input_gradients_with_watermarks, input_gradients_without_watermarks]),
        'Prediction': ['With Watermarks (Correct)'] * len(watermarked_correct) + ['With Watermarks (Wrong)'] *  len(watermarked_wrong) + ['Without Watermarks (Correct)'] * len(not_watermarked_correct) + ['Without Watermarks (Wrong)'] * len(not_watermarked_wrong)
    }

    # Create DataFrame
    data_df = pd.DataFrame(data)

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=4)

    # Violin Plot
    plt.figure(figsize=(10, 12))
    sns.violinplot(data=[watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong], showmedians=True, palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Violin Plot [Input Gradients]', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'violin_by_color.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 12))
    sns.boxplot(data=[watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong], palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Box Plot [Input Gradients]', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'boxplot_by_color.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong],
             bins=20, alpha=0.7, color=colors[:4], label=['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)


    plt.legend()
    plt.title('Histogram [Input Gradients]', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'hist_by_color.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(watermarked_correct, color=colors[0], label='With Watermarks (Correct)')
    sns.kdeplot(watermarked_wrong, color=colors[1], label='With Watermarks (Wrong)')
    sns.kdeplot(not_watermarked_correct, color=colors[2], label='Without Watermarks (Correct)')
    sns.kdeplot(not_watermarked_wrong, color=colors[3], label='Without Watermarks (Wrong)')
    plt.legend()
    plt.title('KDE Plot [Input Gradients]', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'kde_by_color.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 12))
    sns.swarmplot(x=np.concatenate([np.zeros_like(watermarked_correct), np.ones_like(watermarked_wrong),
                                    2*np.ones_like(not_watermarked_correct), 3*np.ones_like(not_watermarked_wrong)]),
                  y=np.concatenate([watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3], ['Watermarked (Correct)', 'Watermarked (Wrong)', 'Not Watermarked (Correct)', 'Not Watermarked (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Swarm Plot [Input Gradients]', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    plt.savefig(os.path.join(arguments_folder, 'swarm_by_color.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 12))
    sns.histplot(data_df, x='Input Gradient Magnitude', cumulative=True, hue='Prediction', element='step', palette=colors)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Input Gradient CDF Plot with Hue [Input Gradients]', fontsize=14)
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='With Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[1], lw=2, label='With Watermarks (Wrong)'),
                   plt.Line2D([0], [0], color=colors[2], lw=2, label='Without Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[3], lw=2, label='Without Watermarks (Wrong)')]

    # Add the legend to the plot
    plt.legend(handles=legend_elements, title='Prediction', title_fontsize=12, loc='lower right')

    plt.savefig(os.path.join(arguments_folder, 'cdf_by_color.pdf'))
    plt.close()

    # Combined
    plt.figure(figsize=(12, 12))

    plt.suptitle('Combined Plots [Input Gradients]', fontsize=16)

    plt.subplot(321)
    sns.violinplot(data=[watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong], showmedians=True, palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    plt.subplot(322)
    sns.boxplot(data=[watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong], palette=colors[:4])
    plt.xticks([0, 1, 2, 3], ['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    plt.subplot(323)
    plt.hist([watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong],
             bins=20, alpha=0.7, color=colors[:4], label=['With Watermarks (Correct)', 'With Watermarks (Wrong)', 'Without Watermarks (Correct)', 'Without Watermarks (Wrong)'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    plt.subplot(324)
    sns.kdeplot(watermarked_correct, color=colors[0], label='With Watermarks (Correct)')
    sns.kdeplot(watermarked_wrong, color=colors[1], label='With Watermarks (Wrong)')
    sns.kdeplot(not_watermarked_correct, color=colors[2], label='Without Watermarks (Correct)')
    sns.kdeplot(not_watermarked_wrong, color=colors[3], label='Without Watermarks (Wrong)')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)

    plt.subplot(325)
    sns.swarmplot(x=np.concatenate([np.zeros_like(watermarked_correct), np.ones_like(watermarked_wrong),
                                    2*np.ones_like(not_watermarked_correct), 3*np.ones_like(not_watermarked_wrong)]),
                  y=np.concatenate([watermarked_correct, watermarked_wrong, not_watermarked_correct, not_watermarked_wrong]),
                  palette=colors)
    plt.xticks([0, 1, 2, 3], ['Watermarked (Correct)', 'Watermarked (Wrong)', 'Not Watermarked (Correct)', 'Not Watermarked (Wrong)'], rotation=20)
    plt.xlabel('')
    plt.title('Swarm Plot', fontsize=14)
    plt.ylabel('Input Gradient Magnitude', fontsize=12)

    plt.subplot(326)
    sns.histplot(data_df, x='Input Gradient Magnitude', cumulative=True, hue='Prediction', element='step', palette=colors)
    plt.xlabel('Input Gradient Magnitude', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Input Gradient CDF Plot with Hue', fontsize=14)
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='With Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[1], lw=2, label='With Watermarks (Wrong)'),
                   plt.Line2D([0], [0], color=colors[2], lw=2, label='Without Watermarks (Correct)'),
                   plt.Line2D([0], [0], color=colors[3], lw=2, label='Without Watermarks (Wrong)')]
    # Add the legend to the plot
    plt.legend(handles=legend_elements, title='Prediction', title_fontsize=12, loc='lower right')

    # Add space between subplots
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_by_color.pdf'))
    plt.close()

    # Calculate correlation coefficients
    input_gradients_correct = np.array(
        [float(item[0][0]) for item in correct_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot],
    )
    input_gradients_wrong = np.array(
        [float(item[0][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_confound.max_ig_label_list_for_score_plot]
    )

    size_min = min([len(input_gradients_correct), len(input_gradients_wrong)])
    pearson_corr, pearson_p_value = stats.pearsonr(input_gradients_correct[:size_min], input_gradients_wrong[:size_min])
    spearman_corr, spearman_p_value = stats.spearmanr(input_gradients_correct[:size_min], input_gradients_wrong[:size_min])

    # Create a DataFrame to store the data
    data_df = pd.DataFrame({
        'Input Gradient Magnitude': np.concatenate([input_gradients_wrong, input_gradients_correct]),
        'Correct': ['No'] * len(input_gradients_wrong) + ['Yes'] * len(input_gradients_correct)
    })

    # Set a common color palette for both groups
    colors = sns.color_palette('Set2', n_colors=2)

    # Set a larger figure size for better visualizations
    plt.figure(figsize=(10, 6))

    # Scatter Plot
    plt.subplot(121)
    sns.scatterplot(data=data_df, x='Input Gradient Magnitude', y='Correct', hue='Correct', palette=colors, legend=False)
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Correct')
    plt.yticks([0, 1], ['No', 'Yes'])

    # Add subtitles for Pearson correlation and p-value
    plt.title(f"Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_p_value:.4f}")

    # Regression Plot
    plt.subplot(122)

    data_df = pd.DataFrame({
        'Input Gradient Magnitude': np.concatenate([input_gradients_correct, input_gradients_wrong]),
        'Correct': ['Correct'] * len(input_gradients_correct) + ['Wrong'] * len(input_gradients_wrong)
    })
    # Convert 'With Watermarks' to 1 and 'Without Watermarks' to 0
    data_df['Correct'] = data_df['Correct'].map({'Correct': 1, 'Wrong': 0})

    sns.regplot(data=data_df, x='Input Gradient Magnitude', y='Correct', logistic=True,
        scatter_kws={'color': colors[0]}, line_kws={'color': colors[1]})
    # invert the y-axis representation
    plt.gca().invert_yaxis()

    plt.title(f"Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p_value:.4f}")
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Correct')
    plt.yticks([0, 1], ['No', 'Yes'])

    # Save the figure
    plt.suptitle('Scatter Plot and Logistic Regression Plot of Input Gradient Magnitude vs. Correct')
    plt.savefig(os.path.join(arguments_folder, 'correlation_plots_correct_wrong.pdf'))
    plt.close()


    # DICTIONARY OF THE PREDICTED INPUT GRADIENT
    # Set a common color palette for all plots
    colors = sns.color_palette('Set2')

    max_arguments_dict = dict()
    max_arguments_dict.update(correct_confound.max_arguments_dict)
    max_arguments_dict.update(correct_not_confound.max_arguments_dict)
    max_arguments_dict.update(wrong_not_confound.max_arguments_dict)
    max_arguments_dict.update(wrong_confound.max_arguments_dict)

    conf_args_dict = dict()
    conf_args_dict.update(correct_confound.max_arguments_dict)
    conf_args_dict.update(wrong_confound.max_arguments_dict)

    # Loop through the superclasses and create a bar plot for each one
    for superclass, subclass_dict in max_arguments_dict.items():
        plt.figure(figsize=(8, 12))  # Adjust the figure size as needed

        # Prepare the data for the bar plot
        subclasses = list(subclass_dict.keys())
        occurrences = [subclass_dict[subc] for subc in subclasses]

        # Create the bar plot
        sns.barplot(x=subclasses, y=occurrences, palette=colors[:len(subclasses)])
        plt.title(f'Number of Occurrences per Subclass in {superclass} [Input Gradients]')
        plt.xlabel('Subclasses')
        plt.ylabel('Number of Occurrences')

        # Rotate the x-axis labels for better visibility
        plt.xticks(rotation=90)

        # Adjust the layout for better visualization
        plt.tight_layout()

        # Save the plot for each superclass
        plt.savefig(os.path.join(arguments_folder, f'{superclass}_barplot.pdf'))


    # Loop through the superclasses and create a bar plot for each one
    for superclass, subclass_dict in conf_args_dict.items():
        plt.figure(figsize=(8, 12))  # Adjust the figure size as needed

        # Prepare the data for the bar plot
        subclasses = list(subclass_dict.keys())
        occurrences = [subclass_dict[subc] for subc in subclasses]

        # Create the bar plot
        sns.barplot(x=subclasses, y=occurrences, palette=colors[:len(subclasses)])
        plt.title(f'Number of Occurrences per Subclass in {superclass} (Watermarked) [Input Gradients]')
        plt.xlabel('Subclasses')
        plt.ylabel('Number of Occurrences')

        # Rotate the x-axis labels for better visibility
        plt.xticks(rotation=90)

        # Adjust the layout for better visualization
        plt.tight_layout()

        # Save the plot for each superclass
        plt.savefig(os.path.join(arguments_folder, f'{superclass}_barplot_conf.pdf'))


    ## PUT TOGETHER THE INPUT AND THE LABEL GRADIENT
    input_gradient_magnitudes = np.array(
        [float(item[0][0]) for item in correct_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_confound.max_ig_label_list_for_score_plot]
    )

    label_scores = np.array(
        [float(item[1][0]) for item in correct_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_confound.max_ig_label_list_for_score_plot]
    )

    # Create a pandas DataFrame
    data_df = pd.DataFrame({
        'Input Gradient Magnitude': input_gradient_magnitudes,
        'Label Score': label_scores}
    )

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=2)

    # Jointplot with Regression
    plt.figure(figsize=(12, 10))
    sns.jointplot(data=data_df, x='Input Gradient Magnitude', y='Label Score', kind='reg', color=colors[1])
    plt.suptitle('Pair Plot: Input Gradients vs Label Score', fontsize=8)

    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_joint_plot.pdf'))
    plt.close()

    # Pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(data_df, vars=['Input Gradient Magnitude', 'Label Score'], palette=colors, kind='scatter')

    plt.suptitle('Pair Plot: Input Gradients vs Label Score', fontsize=8)
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_pair_plot.pdf'))
    plt.close()

    # Violin Plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=data_df[['Input Gradient Magnitude', 'Label Score']], palette=colors)
    plt.title('Violin Plot: Input Gradient and Label Score')

    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_violin_plot.pdf'))
    plt.close()

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap: Input Gradient vs. Label Score')
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_heatmap.pdf'))
    plt.close()

    # Hexbin Plot
    plt.figure(figsize=(8, 6))
    plt.hexbin(input_gradient_magnitudes, label_scores, gridsize=20, cmap='Blues')
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Label Score')
    plt.title('Hexbin Plot: Input Gradient vs. Label Score')
    plt.colorbar(label='Count')
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_hexbin_plot.pdf'))
    plt.close()

    # Scatter Plot with Regression Line
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Input Gradient Magnitude', y='Label Score', data=data_df, color='blue')
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Label Score')
    plt.title('Scatter Plot with Regression Line: Input Gradient vs. Label Score')
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_scatter_with_regression.pdf'))
    plt.close()

    # NOW FOR ALL THE THINGS TOGETHER:

    # Assuming you have the data for each category in the following variables
    input_gradient_magnitudes = np.array(
        [float(item[0][0]) for item in correct_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_confound.max_ig_label_list_for_score_plot]
    )

    label_scores = np.array(
        [float(item[1][0]) for item in correct_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_confound.max_ig_label_list_for_score_plot]
    )

    input_gradients_with_correct_watermarks = np.array([float(item[0][0]) for item in correct_confound.max_ig_label_list_for_score_plot])
    input_gradients_with_wrong_watermarks = np.array([float(item[0][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot])
    input_gradients_with_correct_not_watermarks = np.array([float(item[0][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot])
    input_gradients_with_wrong_not_watermarks = np.array([float(item[0][0]) for item in wrong_confound.max_ig_label_list_for_score_plot])

    label_score_with_correct_watermarks = np.array([float(item[1][0]) for item in correct_confound.max_ig_label_list_for_score_plot])
    label_score_with_wrong_watermarks = np.array([float(item[1][0]) for item in correct_not_confound.max_ig_label_list_for_score_plot])
    label_score_with_correct_not_watermarks = np.array([float(item[1][0]) for item in wrong_not_confound.max_ig_label_list_for_score_plot])
    label_score_with_wrong_not_watermarks = np.array([float(item[1][0]) for item in wrong_confound.max_ig_label_list_for_score_plot])

    # Create DataFrames for each category
    data_with_correct_watermarks = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_correct_watermarks,
        'Label Score': label_score_with_correct_watermarks,
        'Category': 'Watermarked (Correct)'
    })

    data_with_wrong_watermarks = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_wrong_watermarks,
        'Label Score': label_score_with_wrong_watermarks,
        'Category': 'Watermarked (Wrong)'
    })

    data_with_correct_not_watermarks = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_correct_not_watermarks,
        'Label Score': label_score_with_correct_not_watermarks,
        'Category': 'Not Watermarked (Correct)'
    })

    data_with_wrong_not_watermarks = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_wrong_not_watermarks,
        'Label Score': label_score_with_wrong_not_watermarks,
        'Category': 'Not Watermarked (Wrong)'
    })

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=4)

    # Set a larger figure size for better visualizations
    plt.figure(figsize=(10, 8))

    # Scatter Plot
    sns.scatterplot(data=pd.concat([data_with_correct_watermarks, data_with_wrong_watermarks, data_with_correct_not_watermarks, data_with_wrong_not_watermarks]),
                    x='Input Gradient Magnitude', y='Label Score', hue='Category', palette=colors)
    plt.title('Scatter Plot: Input Gradient vs. Label Score')
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Label Score')

    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_scatter_plot_divided.pdf'))
    plt.close()

    # Violin Plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='Category', y='Input Gradient Magnitude', data=pd.concat([data_with_correct_watermarks, data_with_wrong_watermarks, data_with_correct_not_watermarks, data_with_wrong_not_watermarks]), palette=colors)
    plt.title('Violin Plot: Input Gradient Magnitude for Different Cases')
    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_violin_plot_divided.pdf'))
    plt.close()

    # Pairplot
    plt.figure(figsize=(10, 8))
    sns.pairplot(data=pd.concat([data_with_correct_watermarks, data_with_wrong_watermarks, data_with_correct_not_watermarks, data_with_wrong_not_watermarks]),
                 vars=['Input Gradient Magnitude', 'Label Score'], hue='Category', palette=colors, kind='scatter')

    plt.suptitle('Pair Plot: Input Gradient Magnitude for Different Cases', fontsize=8)

    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_pair_plot_divided.pdf'))
    plt.close()

    # Set a larger figure size for better visualizations
    plt.figure(figsize=(10, 8))

    # Create a FacetGrid for the joint plot
    g = sns.FacetGrid(pd.concat([data_with_correct_watermarks, data_with_wrong_watermarks, data_with_correct_not_watermarks, data_with_wrong_not_watermarks]),
                      col='Category', hue='Category', palette=colors, height=5)

    # Scatter Plot in each subplot
    g.map_dataframe(sns.scatterplot, x='Input Gradient Magnitude', y='Label Score')
    g.set_axis_labels('Input Gradient Magnitude', 'Label Score')
    g.set_titles(col_template='Category: {col_name}', fontweight='bold')
    g.add_legend(title='Category', title_fontsize=12, label_order=['Watermarked (Correct)', 'Watermarked (Wrong)', 'Not Watermarked (Correct)', 'Not Watermarked (Wrong)'])
    # Save the joint plot
    plt.suptitle('Pair Plot: Input Gradients vs Label Score', fontsize=12)
    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_joint_plot_divided.pdf'))

    # LABEL GRADIENT
    label_gradients_with_unbalance = np.array(
        [float(item[1][0]) for item in correct_lab.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab.max_ig_label_list_for_score_plot]
    )

    label_gradients_without_unbalance = np.array(
        [float(item[1][0]) for item in corr_no_conf.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_no_conf.max_ig_label_list_for_score_plot]
    )

    lab_gradient_conf_correct=np.array([float(item[1][0]) for item in correct_lab.max_ig_label_list_for_score_plot])
    lab_gradient_wrong=np.array([float(item[1][0]) for item in wrong_lab.max_ig_label_list_for_score_plot])
    not_lab_gradient_conf_correct = np.array([float(item[1][0]) for item in corr_no_conf.max_ig_label_list_for_score_plot])
    not_lab_gradient_conf_wrong = np.array([float(item[1][0]) for item in wrong_no_conf.max_ig_label_list_for_score_plot])
    not_lab_gradient_conf_but_watermar_correct = np.array([float(item[1][0]) for item in correct_confound.max_ig_label_list_for_score_plot])
    not_lab_gradient_conf_but_watermar_wrong = np.array([float(item[1][0]) for item in wrong_confound.max_ig_label_list_for_score_plot])

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2')

    # ALL
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[lab_gradient_conf_correct, lab_gradient_wrong, not_lab_gradient_conf_correct, not_lab_gradient_conf_wrong, not_lab_gradient_conf_but_watermar_correct, not_lab_gradient_conf_but_watermar_wrong],
               showmedians=True, palette=colors)
    plt.xticks([0, 1, 2, 3, 4, 5], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)', 'With Watermark (Correct)', 'With Watermark (Wrong)'], rotation=21)
    plt.xlabel('')
    plt.title('Violin Plot [Label Gradient]', fontsize=14)
    plt.ylabel('Label Gradient Magnitude', fontsize=12)
    plt.tight_layout()

    # Save Violin Plot
    plt.savefig(os.path.join(arguments_folder, 'violin_plot_label_all.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[lab_gradient_conf_correct, lab_gradient_wrong, not_lab_gradient_conf_correct, not_lab_gradient_conf_wrong, not_lab_gradient_conf_but_watermar_correct, not_lab_gradient_conf_but_watermar_wrong],
               palette=colors)
    plt.xticks([0, 1, 2, 3, 4, 5], ['With Sample Imbalance (Correct)', 'With Sample Imbalance (Wrong)', 'Without Sample Imbalance (Correct)', 'Without Sample Imbalance (Wrong)', 'With Watermark (Correct)', 'With Watermark (Wrong)'], rotation=21)
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)
    plt.title('Box Plot [Label Gradient]', fontsize=14)
    plt.tight_layout()

    # Save Box Plot
    plt.savefig(os.path.join(arguments_folder, 'box_plot_label_all.pdf'))
    plt.close()

    # NOT WITH ALL
    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[label_gradients_with_unbalance, label_gradients_without_unbalance],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.xlabel('')
    plt.title('Violin Plot [Label Gradient]', fontsize=14)
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # Save Violin Plot
    plt.savefig(os.path.join(arguments_folder, 'violin_plot_label.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[label_gradients_with_unbalance, label_gradients_without_unbalance],
            palette=colors)
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)
    plt.title('Box Plot [Label Gradient]', fontsize=14)

    # Save Box Plot
    plt.savefig(os.path.join(arguments_folder, 'box_plot_label.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([label_gradients_with_unbalance, label_gradients_without_unbalance],
             bins=20, alpha=0.7, color=colors[:2], label=['With Sample Imbalance', 'Without Sample Imbalance'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.title('Histogram [Label Gradient]', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # Save Histogram
    plt.savefig(os.path.join(arguments_folder, 'histogram_label.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(label_gradients_with_unbalance, color=colors[0], label='With Sample Imbalance')
    sns.kdeplot(label_gradients_without_unbalance, color=colors[1], label='Without Sample Imbalance')
    plt.legend()
    plt.title('KDE Plot [Label Gradient]', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # Save KDE Plot
    plt.savefig(os.path.join(arguments_folder, 'kde_plot_label.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x=np.concatenate([np.zeros_like(label_gradients_with_unbalance), np.ones_like(label_gradients_without_unbalance)]),
                  y=np.concatenate([label_gradients_with_unbalance, label_gradients_without_unbalance]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.title('Swarm Plot [Label Gradient]', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # Save Swarm Plot
    plt.savefig(os.path.join(arguments_folder, 'swarm_plot_label.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 6))
    plt.hist(label_gradients_with_unbalance, density=True, cumulative=True, histtype='step', label='With Sample Imbalance', bins=100, linewidth=2, color=colors[0])
    plt.hist(label_gradients_without_unbalance, density=True, cumulative=True, histtype='step', label='Without Sample Imbalance', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot [Label Gradient]', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save CDF Plot
    plt.savefig(os.path.join(arguments_folder, 'cdf_plot_label.pdf'))
    plt.close()

    # Create a new figure for the combined plots
    plt.figure(figsize=(16, 12))

    # Combined Plots
    plt.suptitle('Combined Plots [Label Gradient]', fontsize=16)

    # Violin Plot
    plt.subplot(231)
    sns.violinplot(data=[label_gradients_with_unbalance, label_gradients_without_unbalance],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # Box Plot
    plt.subplot(232)
    sns.boxplot(data=[label_gradients_with_unbalance, label_gradients_without_unbalance],
            palette=colors)
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    # Histogram
    plt.subplot(233)
    plt.hist([label_gradients_with_unbalance, label_gradients_without_unbalance], bins=20, alpha=0.7, color=colors[:2], label=['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # KDE Plot
    plt.subplot(234)
    sns.kdeplot(label_gradients_with_unbalance, color=colors[0], label='With Sample Imbalance')
    sns.kdeplot(label_gradients_without_unbalance, color=colors[1], label='Without Sample Imbalance')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # Swarm Plot
    plt.subplot(235)
    sns.swarmplot(x=np.concatenate([np.zeros_like(label_gradients_with_unbalance), np.ones_like(label_gradients_without_unbalance)]),
                  y=np.concatenate([label_gradients_with_unbalance, label_gradients_without_unbalance]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Sample Imbalance', 'Without Sample Imbalance'])
    plt.title('Swarm Plot', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # CDF Plot
    plt.subplot(236)
    plt.hist(label_gradients_with_unbalance, density=True, cumulative=True, histtype='step', label='With Sample Imbalance', bins=100, linewidth=2, color=colors[0])
    plt.hist(label_gradients_without_unbalance, density=True, cumulative=True, histtype='step', label='Without Sample Imbalance', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save the combined plot
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_label.pdf'))
    plt.close()

    ## TABLE
    plot_gradient_analysis_table_max(
        wrong_lab=wrong_lab.bucket_list,
        wrong_confound=wrong_confound.bucket_list,
        wrong_lab_img_confound=wrong_lab_img_confound.bucket_list,
        wrong_ok=wrong_not_confound.bucket_list,
        arguments_folder=arguments_folder,
        tau=tau,
    )

    score_barplot(
        correct_confound.prediction_influence_parent_counter
        + wrong_confound.prediction_influence_parent_counter
        + correct_not_confound.prediction_influence_parent_counter
        + wrong_not_confound.prediction_influence_parent_counter
        + correct_lab.prediction_influence_parent_counter
        + wrong_lab.prediction_influence_parent_counter
        + wrong_lab_img_confound.prediction_influence_parent_counter
        + correct_lab_img_confound.prediction_influence_parent_counter,
        correct_confound.prediction_does_not_influence_parent_counter
        + wrong_confound.prediction_does_not_influence_parent_counter
        + correct_not_confound.prediction_does_not_influence_parent_counter
        + wrong_not_confound.prediction_does_not_influence_parent_counter
        + correct_lab.prediction_does_not_influence_parent_counter
        + wrong_lab.prediction_does_not_influence_parent_counter
        + wrong_lab_img_confound.prediction_does_not_influence_parent_counter
        + correct_lab_img_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Influence of Subclass Predictions on Parent Classes",
    )

    score_barplot_list(
        [
            correct_confound.prediction_influence_parent_counter,
            wrong_confound.prediction_influence_parent_counter,
            correct_not_confound.prediction_influence_parent_counter,
            wrong_not_confound.prediction_influence_parent_counter,
            correct_lab.prediction_influence_parent_counter,
            wrong_lab.prediction_influence_parent_counter,
            wrong_lab_img_confound.prediction_influence_parent_counter,
            correct_lab_img_confound.prediction_influence_parent_counter,
        ],
        [
            correct_confound.prediction_does_not_influence_parent_counter,
            wrong_confound.prediction_does_not_influence_parent_counter,
            correct_not_confound.prediction_does_not_influence_parent_counter,
            wrong_not_confound.prediction_does_not_influence_parent_counter,
            correct_lab.prediction_does_not_influence_parent_counter,
            wrong_lab.prediction_does_not_influence_parent_counter,
            wrong_lab_img_confound.prediction_does_not_influence_parent_counter,
            correct_lab_img_confound.prediction_does_not_influence_parent_counter,
        ],
        [
            "Influence [Correct+Confound]",
            "Influence [NotCorrect+Confound]",
            "Influence [Correct+NotConfound]",
            "Influence [NotCorrect+NotConfound]",
            "Influence [Correct+Imbalance]",
            "Influence [NotCorrect+Imbalance]",
            "Influence [NotCorrect+Conf+Imbalance]",
            "Influence [Correct+Conf+Imbalance]",
        ],
        [
            "Not Influence [Correct+Confound]",
            "Not Influence [NotCorrect+Confound]",
            "Not Influence [Correct+NotConfound]",
            "Not Influence [NotCorrect+NotConfound]",
            "Not Influence [Correct+Imbalance]",
            "Not Influence [NotCorrect+Imbalance]",
            "Not Influence [NotCorrect+Conf+Imbalance]",
            "Not Influence [Correct+Conf+Imbalance]",
        ],
        arguments_folder,
        "Influence of Subclass Predictions on Parent Classes",
    )

    # LAB CONFOUNDED CASE:

    input_gradient_magnitudes = np.array(
        [float(item[0][0]) for item in correct_lab.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_lab.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot]
    )

    label_scores = np.array(
        [float(item[1][0]) for item in correct_lab.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot]
    )

    # Create a pandas DataFrame
    data_df = pd.DataFrame({
        'Input Gradient Magnitude': input_gradient_magnitudes,
        'Label Score': label_scores}
    )

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=2)

    # Jointplot with Regression
    plt.figure(figsize=(12, 10))
    sns.jointplot(data=data_df, x='Input Gradient Magnitude', y='Label Score', kind='reg', color=colors[1])
    plt.suptitle('Pair Plot: Input Gradients vs Label Score [Label Confounded Case]', fontsize=8)

    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_joint_plot_lab.pdf'))
    plt.close()

    # Pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(data_df, vars=['Input Gradient Magnitude', 'Label Score'], palette=colors, kind='scatter')

    plt.suptitle('Pair Plot: Input Gradients vs Label Score [Label Confounded Case]', fontsize=8)
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_pair_plot_lab.pdf'))
    plt.close()

    # Violin Plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=data_df[['Input Gradient Magnitude', 'Label Score']], palette=colors)
    plt.title('Violin Plot: Input Gradient and Label Score [Label Confounded Case]')

    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_violin_plot_lab.pdf'))
    plt.close()

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap: Input Gradient vs. Label Score [Label Confounded Case]')
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_heatmap_lab.pdf'))
    plt.close()

    # Hexbin Plot
    plt.figure(figsize=(8, 6))
    plt.hexbin(input_gradient_magnitudes, label_scores, gridsize=20, cmap='Blues')
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Label Score')
    plt.title('Hexbin Plot: Input Gradient vs. Label Score [Label Confounded Case]')
    plt.colorbar(label='Count')
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_hexbin_plot_lab.pdf'))
    plt.close()

    # Scatter Plot with Regression Line
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Input Gradient Magnitude', y='Label Score', data=data_df, color='blue')
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Label Score')
    plt.title('Scatter Plot with Regression Line: Input Gradient vs. Label Score [Label Confounded Case]')
    plt.savefig(os.path.join(arguments_folder, f'gradient_analysis_scatter_with_regression_lab.pdf'))
    plt.close()

    # NOW FOR ALL THE THINGS TOGETHER:

    # Assuming you have the data for each category in the following variables
    input_gradient_magnitudes = np.array(
        [float(item[0][0]) for item in correct_lab.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_lab.max_ig_label_list_for_score_plot] +
        [float(item[0][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot]
    )

    label_scores = np.array(
        [float(item[1][0]) for item in correct_lab.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot]
    )

    input_gradients_with_correct_lab = np.array([float(item[0][0]) for item in correct_lab.max_ig_label_list_for_score_plot])
    input_gradients_with_correct_lab_watermarks = np.array([float(item[0][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot])
    input_gradients_with_wrong_lab = np.array([float(item[0][0]) for item in wrong_lab.max_ig_label_list_for_score_plot])
    input_gradients_with_wrong_lab_watermarks = np.array([float(item[0][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot])

    label_score_with_correct_lab = np.array([float(item[1][0]) for item in correct_lab.max_ig_label_list_for_score_plot])
    label_score_with_correct_lab_watermarks = np.array([float(item[1][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot])
    label_score_with_wrong_lab = np.array([float(item[1][0]) for item in wrong_lab.max_ig_label_list_for_score_plot])
    label_score_wrong_lab_watermarks = np.array([float(item[1][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot])

    # Create DataFrames for each category
    data_with_correct_lab = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_correct_lab,
        'Label Score': label_score_with_correct_lab,
        'Category': 'Label Confound (Correct)'
    })

    data_with_wrong_lab = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_wrong_lab,
        'Label Score': label_score_with_wrong_lab,
        'Category': 'Label Confound (Wrong)'
    })

    data_with_with_correct_lab_watermarks = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_correct_lab_watermarks,
        'Label Score': label_score_with_correct_lab_watermarks,
        'Category': 'Label Confound and Watermarked (Correct)'
    })

    data_with_wrong_lab_watermarks = pd.DataFrame({
        'Input Gradient Magnitude': input_gradients_with_wrong_lab_watermarks,
        'Label Score': label_score_wrong_lab_watermarks,
        'Category': 'Label Confound and Watermarked (Wrong)'
    })

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2', n_colors=4)

    # Set a larger figure size for better visualizations
    plt.figure(figsize=(10, 8))

    # Scatter Plot
    sns.scatterplot(data=pd.concat([data_with_correct_lab, data_with_wrong_lab, data_with_with_correct_lab_watermarks, data_with_wrong_lab_watermarks]),
                    x='Input Gradient Magnitude', y='Label Score', hue='Category', palette=colors)
    plt.title('Scatter Plot: Input Gradient vs. Label Score [Label]')
    plt.xlabel('Input Gradient Magnitude')
    plt.ylabel('Label Score')

    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_scatter_plot_divided_lab.pdf'))
    plt.close()

    # Violin Plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='Category', y='Input Gradient Magnitude', data=pd.concat([data_with_correct_lab, data_with_wrong_lab, data_with_with_correct_lab_watermarks, data_with_wrong_lab_watermarks]), palette=colors)
    plt.title('Violin Plot: Input Gradient Magnitude for Different Cases [Label Confounded Case]')
    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_violin_plot_divided_lab.pdf'))
    plt.close()

    # Pairplot
    plt.figure(figsize=(10, 8))
    sns.pairplot(data=pd.concat([data_with_correct_lab, data_with_wrong_lab, data_with_with_correct_lab_watermarks, data_with_wrong_lab_watermarks]),
                 vars=['Input Gradient Magnitude', 'Label Score'], hue='Category', palette=colors, kind='scatter')

    plt.suptitle('Pair Plot: Input Gradient Magnitude for Different Cases [Label Confounded Case]', fontsize=8)

    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_pair_plot_divided_lab.pdf'))
    plt.close()

    # Set a larger figure size for better visualizations
    plt.figure(figsize=(10, 8))

    # Create a FacetGrid for the joint plot
    g = sns.FacetGrid(pd.concat([data_with_correct_lab, data_with_wrong_lab, data_with_with_correct_lab_watermarks, data_with_wrong_lab_watermarks]),
                      col='Category', hue='Category', palette=colors, height=5)

    # Scatter Plot in each subplot
    g.map_dataframe(sns.scatterplot, x='Input Gradient Magnitude', y='Label Score')
    g.set_axis_labels('Input Gradient Magnitude', 'Label Score')
    g.set_titles(col_template='Category: {col_name}', fontweight='bold')
    g.add_legend(title='Category', title_fontsize=12, label_order=['Watermarked (Correct)', 'Watermarked (Wrong)', 'Not Watermarked (Correct)', 'Not Watermarked (Wrong)'])
    # Save the joint plot
    plt.suptitle('Pair Plot: Input Gradients vs Label Score [Label Confounded Case]', fontsize=12)
    plt.savefig(os.path.join(arguments_folder, 'gradient_analysis_joint_plot_divided_lab.pdf'))

    # LABEL GRADIENT
    label_gradients_with_watermarks = np.array(
        [float(item[1][0]) for item in correct_lab_img_confound.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab_img_confound.max_ig_label_list_for_score_plot]
    )

    label_gradients_without_watermarks = np.array(
        [float(item[1][0]) for item in correct_lab.max_ig_label_list_for_score_plot] +
        [float(item[1][0]) for item in wrong_lab.max_ig_label_list_for_score_plot]
    )

    # Set a common color palette for all plots
    colors = sns.color_palette('Set2')

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[label_gradients_with_watermarks, label_gradients_without_watermarks],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Violin Plot [Label Gradient, Label Confounded Case]', fontsize=14)
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # Save Violin Plot
    plt.savefig(os.path.join(arguments_folder, 'violin_plot_label_lab.pdf'))
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[label_gradients_with_watermarks, label_gradients_without_watermarks],
            palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)
    plt.title('Box Plot [Label Gradient, Label Confounded Case]', fontsize=14)

    # Save Box Plot
    plt.savefig(os.path.join(arguments_folder, 'box_plot_label_lab.pdf'))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist([label_gradients_with_watermarks, label_gradients_without_watermarks],
             bins=20, alpha=0.7, color=colors[:2], label=['With Watermarks', 'Without Watermarks'])
    jitter = 0
    # Add the values of occurrences on top of the bars
    for container in patches:
        for patch in container:
            height = patch.get_height()
            x, width = patch.get_x(), patch.get_width()
            jittered_height = height + np.random.rand() * jitter
            plt.text(x + width / 2., jittered_height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.title('Histogram [Label Gradient, Label Confounded Case]', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # Save Histogram
    plt.savefig(os.path.join(arguments_folder, 'histogram_label_lab.pdf'))
    plt.close()

    # KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(label_gradients_with_watermarks, color=colors[0], label='With Watermarks')
    sns.kdeplot(label_gradients_without_watermarks, color=colors[1], label='Without Watermarks')
    plt.legend()
    plt.title('KDE Plot [Label Gradient, Label Confounded Case]', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # Save KDE Plot
    plt.savefig(os.path.join(arguments_folder, 'kde_plot_label_lab.pdf'))
    plt.close()

    # Swarm Plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x=np.concatenate([np.zeros_like(label_gradients_with_watermarks), np.ones_like(label_gradients_without_watermarks)]),
                  y=np.concatenate([label_gradients_with_watermarks, label_gradients_without_watermarks]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.title('Swarm Plot [Label Gradient, Label Confounded Case]', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # Save Swarm Plot
    plt.savefig(os.path.join(arguments_folder, 'swarm_plot_label_lab.pdf'))
    plt.close()

    # CDF Plot
    plt.figure(figsize=(10, 6))
    plt.hist(label_gradients_with_watermarks, density=True, cumulative=True, histtype='step', label='With Watermarks', bins=100, linewidth=2, color=colors[0])
    plt.hist(label_gradients_without_watermarks, density=True, cumulative=True, histtype='step', label='Without Watermarks', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot [Label Gradient, Label Confounded Case]', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save CDF Plot
    plt.savefig(os.path.join(arguments_folder, 'cdf_plot_label_lab.pdf'))
    plt.close()

    # Create a new figure for the combined plots
    plt.figure(figsize=(16, 12))

    # Combined Plots
    plt.suptitle('Combined Plots [Label Gradient, Label Confounded Case]', fontsize=16)

    # Violin Plot
    plt.subplot(231)
    sns.violinplot(data=[label_gradients_with_watermarks, label_gradients_without_watermarks],
               showmedians=True, palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Violin Plot', fontsize=14)
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # Box Plot
    plt.subplot(232)
    sns.boxplot(data=[label_gradients_with_watermarks, label_gradients_without_watermarks],
            palette=colors)
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.xlabel('')
    plt.title('Box Plot', fontsize=14)

    # Histogram
    plt.subplot(233)
    plt.hist([label_gradients_with_watermarks, label_gradients_without_watermarks], bins=20, alpha=0.7, color=colors[:2], label=['With Watermarks', 'Without Watermarks'])
    plt.legend()
    plt.title('Histogram', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # KDE Plot
    plt.subplot(234)
    sns.kdeplot(label_gradients_with_watermarks, color=colors[0], label='With Watermarks')
    sns.kdeplot(label_gradients_without_watermarks, color=colors[1], label='Without Watermarks')
    plt.legend()
    plt.title('KDE Plot', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)

    # Swarm Plot
    plt.subplot(235)
    sns.swarmplot(x=np.concatenate([np.zeros_like(label_gradients_with_watermarks), np.ones_like(label_gradients_without_watermarks)]),
                  y=np.concatenate([label_gradients_with_watermarks, label_gradients_without_watermarks]),
                  palette=colors[:2], size=3)  # Decrease the size of markers
    plt.xticks([0, 1], ['With Watermarks', 'Without Watermarks'])
    plt.title('Swarm Plot', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Label Gradient Magnitude', fontsize=12)

    # CDF Plot
    plt.subplot(236)
    plt.hist(label_gradients_with_watermarks, density=True, cumulative=True, histtype='step', label='With Watermarks', bins=100, linewidth=2, color=colors[0])
    plt.hist(label_gradients_without_watermarks, density=True, cumulative=True, histtype='step', label='Without Watermarks', bins=100, linewidth=2, color=colors[1])
    plt.title('CDF Plot', fontsize=14)
    plt.xlabel('Label Gradient Magnitude', fontsize=12)
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), borderaxespad=0.)

    # Save the combined plot
    plt.savefig(os.path.join(arguments_folder, 'combined_plots_label_lab.pdf'))
    plt.close()

    scatter_plot_score(
        ig_list_image,
        conf_list_image,
        correct_list_image,
        ig_list_label,
        correct_list_label,
        ig_list_label_and_image,
        correct_list_label_and_image,
        arguments_folder,
        "Max arguments",
    )

    exit(0)

    # MAX end
    print("Secondo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[el for l in correct_confound.ig_lists for el in l[0]],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.ig_lists for el in l[0]
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.ig_lists for el in l[0]
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.ig_lists for el in l[0]
        ],
        ig_list_label_not_corr=[el for l in wrong_lab.ig_lists for el in l[0]],
        ig_list_label_corr=[el for l in correct_lab.ig_lists for el in l[0]],
        ig_list_lab_image_not_corr_conf=[
            el for l in wrong_lab_img_confound.ig_lists for el in l[0]
        ],
        ig_list_lab_image_corr_conf=[
            el for l in correct_lab_img_confound.ig_lists for el in l[0]
        ],
        ig_list_image_corr_conf_counter=len(correct_confound.ig_lists),
        ig_list_image_corr_not_conf_counter=len(correct_not_confound.ig_lists),
        ig_list_image_not_corr_conf_counter=len(wrong_confound.ig_lists),
        ig_list_image_not_corr_not_conf_counter=len(wrong_not_confound.ig_lists),
        ig_list_label_not_corr_counter=len(wrong_lab.ig_lists),
        ig_list_label_corr_counter=len(correct_lab.ig_lists),
        ig_list_lab_image_corr_conf_counter=len(wrong_lab_img_confound.ig_lists),
        ig_list_lab_image_not_corr_conf_counter=len(correct_lab_img_confound.ig_lists),
        folder=arguments_folder,
        prefix="only_input",
    )

    print("Terzo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[el for l in correct_confound.ig_lists for el in l[1]],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.ig_lists for el in l[1]
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.ig_lists for el in l[1]
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.ig_lists for el in l[1]
        ],
        ig_list_label_not_corr=[el for l in wrong_lab.ig_lists for el in l[1]],
        ig_list_label_corr=[el for l in correct_lab.ig_lists for el in l[1]],
        ig_list_lab_image_not_corr_conf=[
            el for l in wrong_lab_img_confound.ig_lists for el in l[1]
        ],
        ig_list_lab_image_corr_conf=[
            el for l in correct_lab_img_confound.ig_lists for el in l[1]
        ],
        ig_list_image_corr_conf_counter=len(correct_confound.ig_lists),
        ig_list_image_corr_not_conf_counter=len(correct_not_confound.ig_lists),
        ig_list_image_not_corr_conf_counter=len(wrong_confound.ig_lists),
        ig_list_image_not_corr_not_conf_counter=len(wrong_not_confound.ig_lists),
        ig_list_label_not_corr_counter=len(wrong_lab.ig_lists),
        ig_list_label_corr_counter=len(correct_lab.ig_lists),
        ig_list_lab_image_corr_conf_counter=len(wrong_lab_img_confound.ig_lists),
        ig_list_lab_image_not_corr_conf_counter=len(correct_lab_img_confound.ig_lists),
        folder=arguments_folder,
        prefix="only_label",
    )

    print("Quarto")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el for l in correct_confound.ig_lists for e in l for el in e
        ],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.ig_lists for e in l for el in e
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.ig_lists for e in l for el in e
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.ig_lists for e in l for el in e
        ],
        ig_list_label_not_corr=[el for l in wrong_lab.ig_lists for e in l for el in e],
        ig_list_label_corr=[el for l in correct_lab.ig_lists for e in l for el in e],
        ig_list_lab_image_not_corr_conf=[
            el for l in wrong_lab_img_confound.ig_lists for e in l for el in e
        ],
        ig_list_lab_image_corr_conf=[
            el for l in correct_lab_img_confound.ig_lists for e in l for el in e
        ],
        ig_list_image_corr_conf_counter=len(correct_confound.ig_lists),
        ig_list_image_corr_not_conf_counter=len(correct_not_confound.ig_lists),
        ig_list_image_not_corr_conf_counter=len(wrong_confound.ig_lists),
        ig_list_image_not_corr_not_conf_counter=len(wrong_not_confound.ig_lists),
        ig_list_label_not_corr_counter=len(wrong_lab.ig_lists),
        ig_list_label_corr_counter=len(correct_lab.ig_lists),
        ig_list_lab_image_corr_conf_counter=len(wrong_lab_img_confound.ig_lists),
        ig_list_lab_image_not_corr_conf_counter=len(correct_lab_img_confound.ig_lists),
        folder=arguments_folder,
        prefix="all",
    )

    conf_list_image = list(
        itertools.chain(
            [True for _ in range(len(correct_confound.ig_lists))],
            [True for _ in range(len(wrong_confound.ig_lists))],
            [False for _ in range(len(correct_not_confound.ig_lists))],
            [False for _ in range(len(wrong_not_confound.ig_lists))],
        )
    )
    correct_list_image = list(
        itertools.chain(
            [True for _ in range(len(correct_confound.ig_lists))],
            [False for _ in range(len(wrong_confound.ig_lists))],
            [True for _ in range(len(correct_not_confound.ig_lists))],
            [False for _ in range(len(wrong_not_confound.ig_lists))],
        )
    )
    ig_list_image = list(
        itertools.chain(
            correct_confound.ig_lists,
            wrong_confound.ig_lists,
            correct_not_confound.ig_lists,
            wrong_not_confound.ig_lists,
        )
    )
    ig_list_label = list(
        itertools.chain(
            correct_lab.ig_lists,
            wrong_lab.ig_lists,
        )
    )
    correct_list_label = list(
        itertools.chain(
            [True for _ in range(len(correct_lab.ig_lists))],
            [False for _ in range(len(wrong_lab.ig_lists))],
        )
    )
    ig_list_label_and_image = list(
        itertools.chain(
            correct_lab_img_confound.ig_lists,
            wrong_lab_img_confound.ig_lists,
        )
    )
    correct_list_label_and_image = list(
        itertools.chain(
            [True for _ in range(len(correct_lab_img_confound.ig_lists))],
            [False for _ in range(len(wrong_lab_img_confound.ig_lists))],
        )
    )
    print("Quinto")
    scatter_plot_score(
        ig_list_image,
        conf_list_image,
        correct_list_image,
        ig_list_label,
        correct_list_label,
        ig_list_label_and_image,
        correct_list_label_and_image,
        arguments_folder,
        "All",
    )

    correct_guesses_conf = [
        True for _ in range(len(correct_confound.max_arguments_list))
    ]
    wrongly_guesses_conf = [
        False for _ in range(len(wrong_confound.max_arguments_list))
    ]
    print("Sesto")
    #  input_gradient_scatter(
    #      correct_guesses=correct_confound.max_arguments_list,
    #      correct_guesses_conf=correct_guesses_conf,
    #      wrongly_guesses=wrong_confound.max_arguments_list,
    #      wrongly_guesses_conf=wrongly_guesses_conf,
    #      folder=arguments_folder,
    #      prefix="max_score",
    #  )

    # ig_lists_wrt_prediction TODO passiamo qua

    print("Settimo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el for l in correct_confound.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_label_not_corr=[
            el for l in wrong_lab.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_label_corr=[
            el for l in correct_lab.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_lab_image_not_corr_conf=[
            el for l in wrong_lab_img_confound.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_lab_image_corr_conf=[
            el for l in correct_lab_img_confound.ig_lists_wrt_prediction for el in l[0]
        ],
        ig_list_image_corr_conf_counter=len(correct_confound.ig_lists_wrt_prediction),
        ig_list_image_corr_not_conf_counter=len(
            correct_not_confound.ig_lists_wrt_prediction
        ),
        ig_list_image_not_corr_conf_counter=len(wrong_confound.ig_lists_wrt_prediction),
        ig_list_image_not_corr_not_conf_counter=len(
            wrong_not_confound.ig_lists_wrt_prediction
        ),
        ig_list_label_not_corr_counter=len(wrong_lab.ig_lists_wrt_prediction),
        ig_list_label_corr_counter=len(correct_lab.ig_lists_wrt_prediction),
        ig_list_lab_image_corr_conf_counter=len(
            wrong_lab_img_confound.ig_lists_wrt_prediction
        ),
        ig_list_lab_image_not_corr_conf_counter=len(
            correct_lab_img_confound.ig_lists_wrt_prediction
        ),
        folder=arguments_folder,
        prefix="wrt_prediction_only_input",
    )

    print("Ottavo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el for l in correct_confound.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_label_not_corr=[
            el for l in wrong_lab.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_label_corr=[
            el for l in correct_lab.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_lab_image_not_corr_conf=[
            el for l in wrong_lab_img_confound.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_lab_image_corr_conf=[
            el for l in correct_lab_img_confound.ig_lists_wrt_prediction for el in l[1]
        ],
        ig_list_image_corr_conf_counter=len(correct_confound.ig_lists_wrt_prediction),
        ig_list_image_corr_not_conf_counter=len(
            correct_not_confound.ig_lists_wrt_prediction
        ),
        ig_list_image_not_corr_conf_counter=len(wrong_confound.ig_lists_wrt_prediction),
        ig_list_image_not_corr_not_conf_counter=len(
            wrong_not_confound.ig_lists_wrt_prediction
        ),
        ig_list_label_not_corr_counter=len(wrong_lab.ig_lists_wrt_prediction),
        ig_list_label_corr_counter=len(correct_lab.ig_lists_wrt_prediction),
        ig_list_lab_image_corr_conf_counter=len(
            wrong_lab_img_confound.ig_lists_wrt_prediction
        ),
        ig_list_lab_image_not_corr_conf_counter=len(
            correct_lab_img_confound.ig_lists_wrt_prediction
        ),
        folder=arguments_folder,
        prefix="wrt_prediction_only_label",
    )

    print("Nono")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el for l in correct_confound.ig_lists_wrt_prediction for e in l for el in e
        ],
        ig_list_image_corr_not_conf=[
            el
            for l in correct_not_confound.ig_lists_wrt_prediction
            for e in l
            for el in e
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.ig_lists_wrt_prediction for e in l for el in e
        ],
        ig_list_image_not_corr_not_conf=[
            el
            for l in wrong_not_confound.ig_lists_wrt_prediction
            for e in l
            for el in e
        ],
        ig_list_label_not_corr=[
            el for l in wrong_lab.ig_lists_wrt_prediction for e in l for el in e
        ],
        ig_list_label_corr=[
            el for l in correct_lab.ig_lists_wrt_prediction for e in l for el in e
        ],
        ig_list_lab_image_not_corr_conf=[
            el
            for l in wrong_lab_img_confound.ig_lists_wrt_prediction
            for e in l
            for el in e
        ],
        ig_list_lab_image_corr_conf=[
            el
            for l in correct_lab_img_confound.ig_lists_wrt_prediction
            for e in l
            for el in e
        ],
        ig_list_image_corr_conf_counter=len(correct_confound.ig_lists_wrt_prediction),
        ig_list_image_corr_not_conf_counter=len(
            correct_not_confound.ig_lists_wrt_prediction
        ),
        ig_list_image_not_corr_conf_counter=len(wrong_confound.ig_lists_wrt_prediction),
        ig_list_image_not_corr_not_conf_counter=len(
            wrong_not_confound.ig_lists_wrt_prediction
        ),
        ig_list_label_not_corr_counter=len(wrong_lab.ig_lists_wrt_prediction),
        ig_list_label_corr_counter=len(correct_lab.ig_lists_wrt_prediction),
        ig_list_lab_image_corr_conf_counter=len(
            wrong_lab_img_confound.ig_lists_wrt_prediction
        ),
        ig_list_lab_image_not_corr_conf_counter=len(
            correct_lab_img_confound.ig_lists_wrt_prediction
        ),
        folder=arguments_folder,
        prefix="wrt_prediction_all",
    )

    conf_list_image = list(
        itertools.chain(
            [True for _ in range(len(correct_confound.ig_lists_wrt_prediction))],
            [True for _ in range(len(wrong_confound.ig_lists_wrt_prediction))],
            [False for _ in range(len(correct_not_confound.ig_lists_wrt_prediction))],
            [False for _ in range(len(wrong_not_confound.ig_lists_wrt_prediction))],
        )
    )
    correct_list_image = list(
        itertools.chain(
            [True for _ in range(len(correct_confound.ig_lists_wrt_prediction))],
            [False for _ in range(len(wrong_confound.ig_lists_wrt_prediction))],
            [True for _ in range(len(correct_not_confound.ig_lists_wrt_prediction))],
            [False for _ in range(len(wrong_not_confound.ig_lists_wrt_prediction))],
        )
    )
    ig_list_image = list(
        itertools.chain(
            correct_confound.ig_lists_wrt_prediction,
            wrong_confound.ig_lists_wrt_prediction,
            correct_not_confound.ig_lists_wrt_prediction,
            wrong_not_confound.ig_lists_wrt_prediction,
        )
    )
    ig_list_label = list(
        itertools.chain(
            correct_lab.ig_lists_wrt_prediction,
            wrong_lab.ig_lists_wrt_prediction,
        )
    )
    correct_list_label = list(
        itertools.chain(
            [True for _ in range(len(correct_lab.ig_lists_wrt_prediction))],
            [False for _ in range(len(wrong_lab.ig_lists_wrt_prediction))],
        )
    )
    ig_list_label_and_image = list(
        itertools.chain(
            correct_lab_img_confound.ig_lists_wrt_prediction,
            wrong_lab_img_confound.ig_lists_wrt_prediction,
        )
    )
    correct_list_label_and_image = list(
        itertools.chain(
            [
                True
                for _ in range(len(correct_lab_img_confound.ig_lists_wrt_prediction))
            ],
            [False for _ in range(len(wrong_lab_img_confound.ig_lists_wrt_prediction))],
        )
    )

    print("Decimo")
    scatter_plot_score(
        ig_list_image,
        conf_list_image,
        correct_list_image,
        ig_list_label,
        correct_list_label,
        ig_list_label_and_image,
        correct_list_label_and_image,
        arguments_folder,
        "Wrt Prediction",
    )

    # ig_lists_wrt_prediction TODO passiamo qua

    print("Undicesimo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el for l in correct_confound.suitable_gradient_full_list for el in l[0]
        ],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.suitable_gradient_full_list for el in l[0]
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.suitable_gradient_full_list for el in l[0]
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.suitable_gradient_full_list for el in l[0]
        ],
        ig_list_label_not_corr=[
            el for l in wrong_lab.suitable_gradient_full_list for el in l[0]
        ],
        ig_list_label_corr=[
            el for l in correct_lab.suitable_gradient_full_list for el in l[0]
        ],
        ig_list_lab_image_not_corr_conf=[
            el
            for l in wrong_lab_img_confound.suitable_gradient_full_list
            for el in l[0]
        ],
        ig_list_lab_image_corr_conf=[
            el
            for l in correct_lab_img_confound.suitable_gradient_full_list
            for el in l[0]
        ],
        ig_list_image_corr_conf_counter=len(
            correct_confound.suitable_gradient_full_list
        ),
        ig_list_image_corr_not_conf_counter=len(
            correct_not_confound.suitable_gradient_full_list
        ),
        ig_list_image_not_corr_conf_counter=len(
            wrong_confound.suitable_gradient_full_list
        ),
        ig_list_image_not_corr_not_conf_counter=len(
            wrong_not_confound.suitable_gradient_full_list
        ),
        ig_list_label_not_corr_counter=len(wrong_lab.suitable_gradient_full_list),
        ig_list_label_corr_counter=len(correct_lab.suitable_gradient_full_list),
        ig_list_lab_image_corr_conf_counter=len(
            wrong_lab_img_confound.suitable_gradient_full_list
        ),
        ig_list_lab_image_not_corr_conf_counter=len(
            correct_lab_img_confound.suitable_gradient_full_list
        ),
        folder=arguments_folder,
        prefix="wrt_groundtruth_only_input",
    )

    print("Dodicesimo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el for l in correct_confound.suitable_gradient_full_list for el in l[1]
        ],
        ig_list_image_corr_not_conf=[
            el for l in correct_not_confound.suitable_gradient_full_list for el in l[1]
        ],
        ig_list_image_not_corr_conf=[
            el for l in wrong_confound.suitable_gradient_full_list for el in l[1]
        ],
        ig_list_image_not_corr_not_conf=[
            el for l in wrong_not_confound.suitable_gradient_full_list for el in l[1]
        ],
        ig_list_label_not_corr=[
            el for l in wrong_lab.suitable_gradient_full_list for el in l[1]
        ],
        ig_list_label_corr=[
            el for l in correct_lab.suitable_gradient_full_list for el in l[1]
        ],
        ig_list_lab_image_not_corr_conf=[
            el
            for l in wrong_lab_img_confound.suitable_gradient_full_list
            for el in l[1]
        ],
        ig_list_lab_image_corr_conf=[
            el
            for l in correct_lab_img_confound.suitable_gradient_full_list
            for el in l[1]
        ],
        ig_list_image_corr_conf_counter=len(
            correct_confound.suitable_gradient_full_list
        ),
        ig_list_image_corr_not_conf_counter=len(
            correct_not_confound.suitable_gradient_full_list
        ),
        ig_list_image_not_corr_conf_counter=len(
            wrong_confound.suitable_gradient_full_list
        ),
        ig_list_image_not_corr_not_conf_counter=len(
            wrong_not_confound.suitable_gradient_full_list
        ),
        ig_list_label_not_corr_counter=len(wrong_lab.suitable_gradient_full_list),
        ig_list_label_corr_counter=len(correct_lab.suitable_gradient_full_list),
        ig_list_lab_image_corr_conf_counter=len(
            wrong_lab_img_confound.suitable_gradient_full_list
        ),
        ig_list_lab_image_not_corr_conf_counter=len(
            correct_lab_img_confound.suitable_gradient_full_list
        ),
        folder=arguments_folder,
        prefix="wrt_groundtruth_only_label",
    )

    print("Tredicesimo")
    box_plot_input_gradients(
        ig_list_image_corr_conf=[
            el
            for l in correct_confound.suitable_gradient_full_list
            for e in l
            for el in e
        ],
        ig_list_image_corr_not_conf=[
            el
            for l in correct_not_confound.suitable_gradient_full_list
            for e in l
            for el in e
        ],
        ig_list_image_not_corr_conf=[
            el
            for l in wrong_confound.suitable_gradient_full_list
            for e in l
            for el in e
        ],
        ig_list_image_not_corr_not_conf=[
            el
            for l in wrong_not_confound.suitable_gradient_full_list
            for e in l
            for el in e
        ],
        ig_list_label_not_corr=[
            el for l in wrong_lab.suitable_gradient_full_list for e in l for el in e
        ],
        ig_list_label_corr=[
            el for l in correct_lab.suitable_gradient_full_list for e in l for el in e
        ],
        ig_list_lab_image_not_corr_conf=[
            el
            for l in wrong_lab_img_confound.suitable_gradient_full_list
            for e in l
            for el in e
        ],
        ig_list_lab_image_corr_conf=[
            el
            for l in correct_lab_img_confound.suitable_gradient_full_list
            for e in l
            for el in e
        ],
        ig_list_image_corr_conf_counter=len(
            correct_confound.suitable_gradient_full_list
        ),
        ig_list_image_corr_not_conf_counter=len(
            correct_not_confound.suitable_gradient_full_list
        ),
        ig_list_image_not_corr_conf_counter=len(
            wrong_confound.suitable_gradient_full_list
        ),
        ig_list_image_not_corr_not_conf_counter=len(
            wrong_not_confound.suitable_gradient_full_list
        ),
        ig_list_label_not_corr_counter=len(wrong_lab.suitable_gradient_full_list),
        ig_list_label_corr_counter=len(correct_lab.suitable_gradient_full_list),
        ig_list_lab_image_corr_conf_counter=len(
            wrong_lab_img_confound.suitable_gradient_full_list
        ),
        ig_list_lab_image_not_corr_conf_counter=len(
            correct_lab_img_confound.suitable_gradient_full_list
        ),
        folder=arguments_folder,
        prefix="wrt_groundtruth_all",
    )

    conf_list_image = list(
        itertools.chain(
            [True for _ in range(len(correct_confound.suitable_gradient_full_list))],
            [True for _ in range(len(wrong_confound.suitable_gradient_full_list))],
            [
                False
                for _ in range(len(correct_not_confound.suitable_gradient_full_list))
            ],
            [False for _ in range(len(wrong_not_confound.suitable_gradient_full_list))],
        )
    )
    correct_list_image = list(
        itertools.chain(
            [True for _ in range(len(correct_confound.suitable_gradient_full_list))],
            [False for _ in range(len(wrong_confound.suitable_gradient_full_list))],
            [
                True
                for _ in range(len(correct_not_confound.suitable_gradient_full_list))
            ],
            [False for _ in range(len(wrong_not_confound.suitable_gradient_full_list))],
        )
    )
    ig_list_image = list(
        itertools.chain(
            correct_confound.suitable_gradient_full_list,
            wrong_confound.suitable_gradient_full_list,
            correct_not_confound.suitable_gradient_full_list,
            wrong_not_confound.suitable_gradient_full_list,
        )
    )
    ig_list_label = list(
        itertools.chain(
            correct_lab.suitable_gradient_full_list,
            wrong_lab.suitable_gradient_full_list,
        )
    )
    correct_list_label = list(
        itertools.chain(
            [True for _ in range(len(correct_lab.suitable_gradient_full_list))],
            [False for _ in range(len(wrong_lab.suitable_gradient_full_list))],
        )
    )
    ig_list_label_and_image = list(
        itertools.chain(
            correct_lab_img_confound.suitable_gradient_full_list,
            wrong_lab_img_confound.suitable_gradient_full_list,
        )
    )
    correct_list_label_and_image = list(
        itertools.chain(
            [
                True
                for _ in range(
                    len(correct_lab_img_confound.suitable_gradient_full_list)
                )
            ],
            [
                False
                for _ in range(len(wrong_lab_img_confound.suitable_gradient_full_list))
            ],
        )
    )
    print("Quattordicesimo")
    scatter_plot_score(
        ig_list_image,
        conf_list_image,
        correct_list_image,
        ig_list_label,
        correct_list_label,
        ig_list_label_and_image,
        correct_list_label_and_image,
        arguments_folder,
        "Wrt Groundtruth All",
    )

    # TODO
    print("Quindicesimo")
    score_barplot(
        correct_confound.influence_parent_counter,
        correct_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [Correct+Conf]",
    )

    score_barplot(
        correct_not_confound.influence_parent_counter,
        correct_not_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [Correct+NotConf]",
    )

    score_barplot(
        wrong_not_confound.influence_parent_counter,
        wrong_not_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [NotCorrect+NotConf]",
    )

    score_barplot(
        wrong_confound.influence_parent_counter,
        wrong_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [NotCorrect+Conf]",
    )

    score_barplot(
        wrong_not_confound.influence_parent_counter,
        wrong_not_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [NotCorrect+NotConf]",
    )

    score_barplot(
        correct_lab.influence_parent_counter,
        correct_lab.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [Correct+Imbalance]",
    )

    score_barplot(
        wrong_lab.influence_parent_counter,
        wrong_lab.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [NotCorrect+Imbalance]",
    )

    score_barplot(
        wrong_lab_img_confound.influence_parent_counter,
        wrong_lab_img_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [NotCorrect+Conf+Imbalance]",
    )

    score_barplot(
        correct_lab_img_confound.influence_parent_counter,
        correct_lab_img_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence [Correct+Conf+Imbalance]",
    )

    score_barplot(
        correct_confound.influence_parent_counter
        + wrong_confound.influence_parent_counter
        + correct_not_confound.influence_parent_counter
        + wrong_not_confound.influence_parent_counter
        + correct_lab.influence_parent_counter
        + wrong_lab.influence_parent_counter
        + wrong_lab_img_confound.influence_parent_counter
        + correct_lab_img_confound.influence_parent_counter,
        correct_confound.not_influence_parent_counter
        + wrong_confound.not_influence_parent_counter
        + correct_not_confound.not_influence_parent_counter
        + wrong_not_confound.not_influence_parent_counter
        + correct_lab.not_influence_parent_counter
        + wrong_lab.not_influence_parent_counter
        + wrong_lab_img_confound.not_influence_parent_counter
        + correct_lab_img_confound.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Does influence parent prediction vs does not influence",
    )

    score_barplot_list(
        [
            correct_confound.influence_parent_counter,
            wrong_confound.influence_parent_counter,
            correct_not_confound.influence_parent_counter,
            wrong_not_confound.influence_parent_counter,
            correct_lab.influence_parent_counter,
            wrong_lab.influence_parent_counter,
            wrong_lab_img_confound.influence_parent_counter,
            correct_lab_img_confound.influence_parent_counter,
        ],
        [
            correct_confound.not_influence_parent_counter,
            wrong_confound.not_influence_parent_counter,
            correct_not_confound.not_influence_parent_counter,
            wrong_not_confound.not_influence_parent_counter,
            correct_lab.not_influence_parent_counter,
            wrong_lab.not_influence_parent_counter,
            wrong_lab_img_confound.not_influence_parent_counter,
            correct_lab_img_confound.not_influence_parent_counter,
        ],
        [
            "Influence [Correct+Confound]",
            "Influence [NotCorrect+Confound]",
            "Influence [Correct+NotConfound]",
            "Influence [NotCorrect+NotConfound]",
            "Influence [Correct+Imbalance]",
            "Influence [NotCorrect+Imbalance]",
            "Influence [NotCorrect+Conf+Imbalance]",
            "Influence [Correct+Conf+Imbalance]",
        ],
        [
            "Not Influence [Correct+Confound]",
            "Not Influence [NotCorrect+Confound]",
            "Not Influence [Correct+NotConfound]",
            "Not Influence [NotCorrect+NotConfound]",
            "Not Influence [Correct+Imbalance]",
            "Not Influence [NotCorrect+Imbalance]",
            "Not Influence [NotCorrect+Conf+Imbalance]",
            "Not Influence [Correct+Conf+Imbalance]",
        ],
        arguments_folder,
        "Influence parent prediction vs not influence",
    )

    # TODO

    score_barplot(
        correct_confound.prediction_influence_parent_counter,
        correct_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[On prediction] Does influence parent prediction vs does not influence [Correct+Conf]",
    )

    score_barplot(
        correct_not_confound.prediction_influence_parent_counter,
        correct_not_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [Correct+NotConf]",
    )

    score_barplot(
        wrong_not_confound.prediction_influence_parent_counter,
        wrong_not_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [NotCorrect+NotConf]",
    )

    score_barplot(
        wrong_confound.prediction_influence_parent_counter,
        wrong_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [NotCorrect+Conf]",
    )

    score_barplot(
        wrong_not_confound.prediction_influence_parent_counter,
        wrong_not_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [NotCorrect+NotConf]",
    )

    score_barplot(
        correct_lab.prediction_influence_parent_counter,
        correct_lab.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [Correct+Imbalance]",
    )

    score_barplot(
        wrong_lab.prediction_influence_parent_counter,
        wrong_lab.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [NotCorrect+Imbalance]",
    )

    score_barplot(
        wrong_lab_img_confound.prediction_influence_parent_counter,
        wrong_lab_img_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [NotCorrect+Conf+Imbalance]",
    )

    score_barplot(
        correct_lab_img_confound.prediction_influence_parent_counter,
        correct_lab_img_confound.prediction_does_not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "[Prediction] Does influence parent prediction vs does not influence [Correct+Conf+Imbalance]",
    )

    #  max_arg_dictionary = correct_confound.max_arguments_dict

    #  plot_most_frequent_explainations(
    #      correct_confound.max_arguments_dict, "Max chosen per class: [Correct+Conf]", arguments_folder
    #  )

    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, wrong_confound.max_arguments_dict
    #  )
    #
    plot_most_frequent_explainations(
        wrong_confound.max_arguments_dict,
        "Max chosen per class: [NotCorrect+Conf]",
        arguments_folder,
    )
    #
    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, correct_not_confound.max_arguments_dict
    #  )
    #
    #  plot_most_frequent_explainations(
    #      correct_not_confound.max_arguments_dict, "Max chosen per class: [Correct+NotConf]", arguments_folder
    #  )
    #
    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, wrong_not_confound.max_arguments_dict
    #  )
    #
    #  plot_most_frequent_explainations(
    #      wrong_not_confound.max_arguments_dict, "Max chosen per class: [NotCorrect+NotConf]", arguments_folder
    #  )
    #
    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, correct_lab.max_arguments_dict
    #  )
    #
    #  plot_most_frequent_explainations(
    #      correct_lab.max_arguments_dict, "Max chosen per class: [Correct+Imbalance]", arguments_folder
    #  )
    #
    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, wrong_lab.max_arguments_dict
    #  )
    plot_most_frequent_explainations(
        wrong_lab.max_arguments_dict,
        "Max chosen per class: [NotCorrect+Imbalance]",
        arguments_folder,
    )
    #
    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, wrong_lab_img_confound.max_arguments_dict
    #  )
    #
    plot_most_frequent_explainations(
        wrong_lab_img_confound.max_arguments_dict,
        "Max chosen per class: [NotCorrect+Conf+Imbalance]",
        arguments_folder,
    )
    #
    #  max_arg_dictionary = sum_merge_dictionary(
    #      max_arg_dictionary, correct_lab_img_confound.max_arguments_dict
    #  )
    #
    #  plot_most_frequent_explainations(
    #      correct_lab_img_confound.max_arguments_dict, "Max chosen per class: [Correct+Conf+Imbalance]", arguments_folder
    #  )
    #
    #  plot_most_frequent_explainations(
    #      max_arg_dictionary, "Max chosen per class", arguments_folder
    #  )

    #
    #  plot_gradient_analysis_table_full(
    #      wrong_lab=wrong_lab.bucket_list,
    #      wrong_confound=wrong_confound.bucket_list,
    #      wrong_lab_img_confound=wrong_lab_img_confound.bucket_list,
    #      wrong_ok=wrong_not_confound.bucket_list,
    #      arguments_folder=arguments_folder,
    #      tau=tau
    #  )
    #
    plot_gradient_analysis_table_wrt_groundtruth(
        wrong_lab=wrong_lab.bucket_list,
        wrong_confound=wrong_confound.bucket_list,
        wrong_lab_img_confound=wrong_lab_img_confound.bucket_list,
        wrong_ok=wrong_not_confound.bucket_list,
        arguments_folder=arguments_folder,
        tau=tau,
    )

    plot_gradient_analysis_table_wrt_prediction(
        wrong_lab=wrong_lab.bucket_list,
        wrong_confound=wrong_confound.bucket_list,
        wrong_lab_img_confound=wrong_lab_img_confound.bucket_list,
        wrong_ok=wrong_not_confound.bucket_list,
        arguments_folder=arguments_folder,
        tau=tau,
    )


def plot_gradient_analysis_table_wrt_groundtruth(
    wrong_lab: List[ArgumentBucket],
    wrong_confound: List[ArgumentBucket],
    wrong_lab_img_confound: List[ArgumentBucket],
    wrong_ok: List[ArgumentBucket],
    arguments_folder: str,
    tau: float,
):

    # TODO considerando valore massimo e considerando tutti

    columns_header = ["# X > t", "# Y > t", "# X > # Y", "#"]
    rows_header = [
        "Not correct: confound on X",
        "Not correct: confound on Y",
        "Not correct: confound on both XY",
        "Not correct: no confound",
    ]

    # data
    data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    # colors
    # lightgreen, lightcoral
    colors = [
        ["lightgreen", "w", "lightgreen", "w"],
        ["w", "lightgreen", "lightcoral", "w"],
        ["w", "w", "lightcoral", "w"],
        ["w", "w", "w", "w"],
    ]

    for item in wrong_confound:
        ig, ig_label = item.get_gradents_list_separated_by_class(
            item.groundtruth_children
        )
        max_ig = max(ig)
        max_label = max(ig_label)
        if max_ig > tau:
            data[0][0] += 1
        if max_label > tau:
            data[0][1] += 1
        if max_ig > max_label:
            data[0][2] += 1
        data[0][3] += 1

    for item in wrong_lab:
        ig, ig_label = item.get_gradents_list_separated_by_class(
            item.groundtruth_children
        )
        max_ig = max(ig)
        max_label = max(ig_label)
        if max_ig > tau:
            data[1][0] += 1
        if max_label > tau:
            data[1][1] += 1
        if max_ig > max_label:
            data[1][2] += 1
        data[1][3] += 1

    for item in wrong_lab_img_confound:
        ig, ig_label = item.get_gradents_list_separated_by_class(
            item.groundtruth_children
        )
        max_ig = max(ig)
        max_label = max(ig_label)
        if max_ig > tau:
            data[2][0] += 1
        if max_label > tau:
            data[2][1] += 1
        if max_ig > max_label:
            data[2][2] += 1
        data[2][3] += 1

    for item in wrong_ok:
        ig, ig_label = item.get_gradents_list_separated_by_class(
            item.groundtruth_children
        )
        max_ig = max(ig)
        max_label = max(ig_label)
        if max_ig > tau:
            data[3][0] += 1
        if max_label > tau:
            data[3][1] += 1
        if max_ig > max_label:
            data[3][2] += 1
        data[3][3] += 1

    fig, ax1 = plt.subplots(figsize=(10, 2 + len(data) / 2.5))

    rcolors = np.full(len(rows_header), "linen")
    ccolors = np.full(len(columns_header), "lavender")

    table = ax1.table(
        cellText=data,
        cellLoc="center",
        rowLabels=rows_header,
        rowColours=rcolors,
        rowLoc="center",
        colColours=ccolors,
        cellColours=colors,
        colLabels=columns_header,
        loc="center",
    )

    table.scale(1, 2)
    table.set_fontsize(16)
    ax1.axis("off")
    title = f"Gradient Analysis (per samples: wrt groundtruth) Table: tau = {tau}"
    ax1.set_title(f"{title}", weight="bold", size=14, color="k")
    fig.savefig("{}/{}.pdf".format(arguments_folder, title), bbox_inches="tight")
    plt.close(fig)


def plot_gradient_analysis_table_wrt_prediction(
    wrong_lab: List[ArgumentBucket],
    wrong_confound: List[ArgumentBucket],
    wrong_lab_img_confound: List[ArgumentBucket],
    wrong_ok: List[ArgumentBucket],
    arguments_folder: str,
    tau: float,
):

    # TODO considerando valore massimo e considerando tutti

    columns_header = ["# X > t", "# Y > t", "# X > # Y", "#"]
    rows_header = [
        "Not correct: confound on X",
        "Not correct: confound on Y",
        "Not correct: confound on both XY",
        "Not correct: no confound",
    ]

    # data
    data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    # colors
    # lightgreen, lightcoral
    colors = [
        ["lightgreen", "w", "lightgreen", "w"],
        ["w", "lightgreen", "lightcoral", "w"],
        ["w", "w", "lightcoral", "w"],
        ["w", "w", "w", "w"],
    ]

    for item in wrong_confound:
        ig, ig_label = list(), list()
        performed_predictions = (item.prediction == True).nonzero().flatten().tolist()
        for pred_el in performed_predictions:
            if pred_el > 4:
                list_ig, list_label_ig = item.get_gradents_list_separated_by_class(
                    pred_el
                )
                for l_i in list_ig:
                    ig.append(l_i)
                for l_i in list_label_ig:
                    ig_label.append(l_i)

        if not ig or not ig_label:
            print("continue...")
            continue

        max_ig = max(ig)
        max_label = max(ig_label)

        if max_ig > tau:
            data[0][0] += 1
        if max_label > tau:
            data[0][1] += 1
        if max_ig > max_label:
            data[0][2] += 1
        data[0][3] += 1

    for item in wrong_lab:
        ig, ig_label = list(), list()
        performed_predictions = (item.prediction == True).nonzero().flatten().tolist()
        for pred_el in performed_predictions:
            if pred_el > 4:
                list_ig, list_label_ig = item.get_gradents_list_separated_by_class(
                    pred_el
                )
                for l_i in list_ig:
                    ig.append(l_i)
                for l_i in list_label_ig:
                    ig_label.append(l_i)

        if not ig or not ig_label:
            print("continue...")
            continue

        max_ig = max(ig)
        max_label = max(ig_label)

        if max_ig > tau:
            data[1][0] += 1
        if max_label > tau:
            data[1][1] += 1
        if max_ig > max_label:
            data[1][2] += 1
        data[1][3] += 1

    for item in wrong_lab_img_confound:
        ig, ig_label = list(), list()
        performed_predictions = (item.prediction == True).nonzero().flatten().tolist()
        for pred_el in performed_predictions:
            if pred_el > 4:
                list_ig, list_label_ig = item.get_gradents_list_separated_by_class(
                    pred_el
                )
                for l_i in list_ig:
                    ig.append(l_i)
                for l_i in list_label_ig:
                    ig_label.append(l_i)

        if not ig or not ig_label:
            print("continue...")
            continue

        max_ig = max(ig)
        max_label = max(ig_label)

        if max_ig > tau:
            data[2][0] += 1
        if max_label > tau:
            data[2][1] += 1
        if max_ig > max_label:
            data[2][2] += 1
        data[2][3] += 1

    for item in wrong_ok:
        ig, ig_label = list(), list()
        performed_predictions = (item.prediction == True).nonzero().flatten().tolist()
        for pred_el in performed_predictions:
            if pred_el > 4:
                list_ig, list_label_ig = item.get_gradents_list_separated_by_class(
                    pred_el
                )
                for l_i in list_ig:
                    ig.append(l_i)
                for l_i in list_label_ig:
                    ig_label.append(l_i)

        if not ig or not ig_label:
            print("continue...")
            continue

        max_ig = max(ig)
        max_label = max(ig_label)

        if max_ig > tau:
            data[3][0] += 1
        if max_label > tau:
            data[3][1] += 1
        if max_ig > max_label:
            data[3][2] += 1
        data[3][3] += 1

    fig, ax1 = plt.subplots(figsize=(10, 2 + len(data) / 2.5))

    rcolors = np.full(len(rows_header), "linen")
    ccolors = np.full(len(columns_header), "lavender")

    table = ax1.table(
        cellText=data,
        cellLoc="center",
        rowLabels=rows_header,
        rowColours=rcolors,
        rowLoc="center",
        colColours=ccolors,
        cellColours=colors,
        colLabels=columns_header,
        loc="center",
    )

    table.scale(1, 2)
    table.set_fontsize(16)
    ax1.axis("off")
    title = f"Gradient Analysis (per samples: wrt prediction) Table: tau = {tau}"
    ax1.set_title(f"{title}", weight="bold", size=14, color="k")
    fig.savefig("{}/{}.pdf".format(arguments_folder, title), bbox_inches="tight")
    plt.close(fig)


def plot_gradient_analysis_table_max(
    wrong_lab: List[ArgumentBucket],
    wrong_confound: List[ArgumentBucket],
    wrong_lab_img_confound: List[ArgumentBucket],
    wrong_ok: List[ArgumentBucket],
    arguments_folder: str,
    tau: float,
):

    # TODO considerando valore massimo e considerando tutti

    columns_header = ["# X > t", "# Y > t", "# X > # Y", "#"]
    rows_header = [
        "Not correct: confound on X",
        "Not correct: confound on Y",
        "Not correct: confound on both XY",
        "Not correct: no confound",
    ]

    # data
    data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    # colors
    # lightgreen, lightcoral
    colors = [
        ["lightgreen", "w", "lightgreen", "w"],
        ["w", "lightgreen", "lightcoral", "w"],
        ["w", "w", "lightcoral", "w"],
        ["w", "w", "w", "w"],
    ]

    for item in wrong_confound:
        max_ig = item.get_maximum_ig_score()
        max_label = item.get_maximum_label_score()
        if max_ig > tau:
            data[0][0] += 1
        if max_label > tau:
            data[0][1] += 1
        if max_ig > max_label:
            data[0][2] += 1
        data[0][3] += 1

    for item in wrong_lab:
        max_ig = item.get_maximum_ig_score()
        max_label = item.get_maximum_label_score()
        if max_ig > tau:
            data[1][0] += 1
        if max_label > tau:
            data[1][1] += 1
        if max_ig > max_label:
            data[1][2] += 1
        data[1][3] += 1

    for item in wrong_lab_img_confound:
        max_ig = item.get_maximum_ig_score()
        max_label = item.get_maximum_label_score()
        if max_ig > tau:
            data[2][0] += 1
        if max_label > tau:
            data[2][1] += 1
        if max_ig > max_label:
            data[2][2] += 1
        data[2][3] += 1

    for item in wrong_ok:
        max_ig = item.get_maximum_ig_score()
        max_label = item.get_maximum_label_score()
        if max_ig > tau:
            data[3][0] += 1
        if max_label > tau:
            data[3][1] += 1
        if max_ig > max_label:
            data[3][2] += 1
        data[3][3] += 1

    fig, ax1 = plt.subplots(figsize=(10, 2 + len(data) / 2.5))

    rcolors = np.full(len(rows_header), "linen")
    ccolors = np.full(len(columns_header), "lavender")

    table = ax1.table(
        cellText=data,
        cellLoc="center",
        rowLabels=rows_header,
        rowColours=rcolors,
        rowLoc="center",
        colColours=ccolors,
        cellColours=colors,
        colLabels=columns_header,
        loc="center",
    )

    table.scale(1, 2)
    table.set_fontsize(16)
    ax1.axis("off")
    title = f"[Max] Gradient Analysis (per samples) Table: tau = {tau}"
    ax1.set_title(f"{title}", weight="bold", size=14, color="k")
    fig.savefig("{}/{}.pdf".format(arguments_folder, title), bbox_inches="tight")
    plt.close(fig)


def plot_gradient_analysis_table_full(
    wrong_lab: List[ArgumentBucket],
    wrong_confound: List[ArgumentBucket],
    wrong_lab_img_confound: List[ArgumentBucket],
    wrong_ok: List[ArgumentBucket],
    arguments_folder: str,
    tau: float,
):

    # TODO considerando valore massimo e considerando tutti

    columns_header = ["# X > t", "# Y > t", "# X > # Y", "#ig", "#labg"]
    rows_header = [
        "Not correct: confound on X",
        "Not correct: confound on Y",
        "Not correct: confound on both XY",
        "Not correct: no confound",
    ]

    # data
    data = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    # colors
    # lightgreen, lightcoral
    colors = [
        ["lightgreen", "w", "lightgreen", "w", "w"],
        ["w", "lightgreen", "lightcoral", "w", "w"],
        ["w", "w", "lightcoral", "w", "w"],
        ["w", "w", "w", "w", "w"],
    ]

    for item in wrong_confound:
        ig_list, label_g_list = item.get_gradents_list_separated()
        for el_g in ig_list:
            if el_g > tau:
                data[0][0] += 1
        for el_l_g in label_g_list:
            if el_l_g > tau:
                data[0][1] += 1
        for el_l_g in label_g_list:
            for el_g in ig_list:
                if el_g > el_l_g:
                    data[0][2] += 1
        data[0][3] += len(ig_list)
        data[0][4] += len(label_g_list)

    for item in wrong_lab:
        ig_list, label_g_list = item.get_gradents_list_separated()
        for el_g in ig_list:
            if el_g > tau:
                data[1][0] += 1
        for el_l_g in label_g_list:
            if el_l_g > tau:
                data[1][1] += 1
        for el_l_g in label_g_list:
            for el_g in ig_list:
                if el_g > el_l_g:
                    data[1][2] += 1
        data[1][3] += len(ig_list)
        data[1][4] += len(label_g_list)

    for item in wrong_lab_img_confound:
        ig_list, label_g_list = item.get_gradents_list_separated()
        for el_g in ig_list:
            if el_g > tau:
                data[2][0] += 1
        for el_l_g in label_g_list:
            if el_l_g > tau:
                data[2][1] += 1
        for el_l_g in label_g_list:
            for el_g in ig_list:
                if el_g > el_l_g:
                    data[2][2] += 1
        data[2][3] += len(ig_list)
        data[2][4] += len(label_g_list)

    for item in wrong_ok:
        ig_list, label_g_list = item.get_gradents_list_separated()
        for el_g in ig_list:
            if el_g > tau:
                data[3][0] += 1
        for el_l_g in label_g_list:
            if el_l_g > tau:
                data[3][1] += 1
        for el_l_g in label_g_list:
            for el_g in ig_list:
                if el_g > el_l_g:
                    data[3][2] += 1
        data[3][3] += len(ig_list)
        data[3][4] += len(label_g_list)

    fig, ax1 = plt.subplots(figsize=(10, 2 + len(data) / 2.5))

    rcolors = np.full(len(rows_header), "linen")
    ccolors = np.full(len(columns_header), "lavender")

    table = ax1.table(
        cellText=data,
        cellLoc="center",
        rowLabels=rows_header,
        rowColours=rcolors,
        rowLoc="center",
        colColours=ccolors,
        cellColours=colors,
        colLabels=columns_header,
        loc="center",
    )

    table.scale(1, 2)
    table.set_fontsize(16)
    ax1.axis("off")
    title = f"Gradient Analysis (all gradients) Table: tau = {tau}"
    ax1.set_title(f"{title}", weight="bold", size=14, color="k")
    fig.savefig("{}/{}.pdf".format(arguments_folder, title), bbox_inches="tight")
    plt.close(fig)


def sum_merge_dictionary(
    first: Dict[str, Dict[str, int]], second: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """Method which performs a dictionary merge between two dictionaries by summing thbe common values
    Args:
        first [Dict[str, Dict[str, int]]]: first dictionary
        second [Dict[str, Dict[str, int]]]: second dictionary
    """
    new_dict: Dict[str, Dict[str, int]] = first.copy()
    # add overlapped elements if it is necessary
    for key in first:
        if key not in second:
            continue
        for subkey in first[key]:
            if subkey not in second[key]:
                continue
            new_dict[key][subkey] += second[key][subkey]

    # loop over the second dictionary in order to add those lements which have been skipped
    for key in second:
        if key in first:
            continue
        new_dict[key] = second[key]

    # returned the newly created dictionary
    return new_dict


def arguments_step(
    net: nn.Module,
    dataset: str,
    dataloaders: Dict[str, Any],
    device: str,
    force_prediction: bool,
    use_softmax: bool,
    prediction_treshold: float,
    correct_samples_only: bool,
    confounded_samples_only: bool,
    num_element_to_analyze: int,
    labels_name: List[str],
    number_element_to_show: int,
    arguments_folder: str,
    label_loader: bool,
    norm_exponent: int,
    multiply_by_probability_for_label_gradient: bool,
    cincer: bool,
    use_probabilistic_circuits,
    gate: DenseGatingFunction,
    cmpe: CircuitMPE,
    use_gate_output: bool,
) -> ArgumentsStepArgs:
    """Arguments step
    Args:
        net [nn.Module]: neural network
        dataset [str]: dataset name
        dataloaders [Dict[str, Any]: dataloader
        device [str]: device
        force_prediction [bool]: force prediction
        use_softmax [bool]: use softmax
        prediction_treshold [float]: prediction threshold
        correct_samples_only [bool]: correct_samples_only
        confounded_samples_only [bool]: confounded_samples_only
        num_element_to_analyze [int]: number of elements to analyze
        labels_name [List[str]]: list of label names
        number_element_to_show [int]: number elements to show
        arguments_folder [str]: arguments folder
        label_loader [bool]: whether to use the label loader
        nrom_exponent [int]: exponent to use for the norm
    Returns:
        ArgumentsStepArgs: arguments as class
    """

    """List of buckets per each sample"""
    bucket_list: List[ArgumentBucket] = list()
    """
    Table correlation dictionary for each class which contains a dictionary for each arguments name
    which contains a tuple representing whether the same has been correcly guessed and the score associated
    """
    table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]] = dict()
    """List of input gradients explainations which is taken from the explainations which should be the 'right' ones"""
    suitable_ig_explaination_list: List[float] = list()
    """Lists of gradients for each sample, where the first list is the list of input gradients while the second list concerns the label gradient"""
    ig_lists: List[Tuple[List[float], List[float]]] = list()
    """Lists of gradients for each sample, where the first list is the list of input gradients while the second list concerns the label gradient. This concerns only the 'right' classes"""
    suitable_gradient_full_list: List[Tuple[List[float], List[float]]] = list()
    """List of arguments which are the ones with highest score"""
    max_arguments_list: List[float] = list()
    """Maximum ig label list"""
    max_ig_label_list_for_score_plot: List[Tuple[List[float], List[float]]] = list()
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the input gradient]"""
    max_arguments_dict: Dict[str, Dict[str, float]] = dict()
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the label gradient]"""
    label_args_dict: Dict[str, Dict[str, List[float]]] = dict()
    """Counter for showing the single element example"""
    show_element_counter: int = 0
    """Counter which defines how many times a subclass has influenced a superclass"""
    influence_parent_counter: int = 0
    """Counter which defines how many times a subclass has not influenced a superclass"""
    not_influence_parent_counter: int = 0
    """Counter which defined how many times the 'right' argument is the maximuum one"""
    suitable_is_max: int = 0
    """Counter which defined how many times the 'right' argument is not the maximuum one"""
    suitable_is_not_max: int = 0
    """Counter for the prediction"""
    prediction_influence_parent_counter: int = 0
    prediction_does_not_influence_parent_counter: int = 0
    # get_gradents_list_separated_by_class
    ig_lists_wrt_prediction: List[Tuple[List[float], List[float]]] = list()
    """Label entropy"""
    label_entropy_list: List[float] = list()
    """Label entropy"""
    ig_entropy_list: List[List[float]] = list()

    # if label_loader is active then shiwtch to the right dataloader
    loader = dataloaders["test_loader_with_labels_name"]
    if label_loader:
        loader = dataloaders["test_loader_only_label_confounders_with_labels_names"]

    # get samples
    selected_samples = get_samples_for_arguments(
        net=net,
        dataset=dataset,
        dataloaders=dataloaders,
        device=device,
        force_prediction=force_prediction,
        use_softmax=use_softmax,
        prediction_treshold=prediction_treshold,
        loader=loader,
        correct=correct_samples_only,
        num_element_to_analyze=num_element_to_analyze,
        test=dataloaders["test"],
        confounded=confounded_samples_only,
        phase="test",
        use_probabilistic_circuits=use_probabilistic_circuits,
        gate=gate,
        cmpe=cmpe,
    )

    # loop over the samples
    for idx, (c_g, pred, groundtruth) in enumerate(selected_samples):
        # get the bucket of arguments
        bucket = ArgumentBucket(
            c_g,  # sample
            net,  # network
            labels_name,  # label names
            dataloaders["test"].to_eval,  # what to evaluate
            dataloaders["train_R"],  # train R
            pred,  # predicted integer labels
            groundtruth[0],  # parent label
            groundtruth[1],  # children label
            norm_exponent,
            prediction_treshold,
            force_prediction,
            use_softmax,
            multiply_by_probability_for_label_gradient,
            cincer,
            use_probabilistic_circuits,
            gate,
            cmpe,
            use_gate_output,
        )

        # label_entropy_list -> TODO just inserted
        label_entropy_list.append(bucket.get_predicted_label_entropy())
        ig_entropy_list.append(bucket.get_ig_entropy())

        # ig lists
        ig_lists.append(bucket.get_gradents_list_separated())

        # TODO provare questo
        performed_predictions = (bucket.prediction == True).nonzero().flatten().tolist()
        for pred_el in performed_predictions:
            if pred_el > 4:
                ig_lists_wrt_prediction.append(
                    bucket.get_gradents_list_separated_by_class(pred_el)
                )

        # table correlation
        table_correlation = get_table_correlation_dictionary_from(
            bucket, correct_samples_only, table_correlation
        )
        # get the bucket in the bucket list
        bucket_list.append(bucket)
        # maximum score coming from the input gradienst
        max_score_ig: float = bucket.get_maximum_ig_score()

        # get the groundtruth label and the groundtruth gradient and score
        ground_truth_lab, ig_grad_score = bucket.get_ig_groundtruth()
        # add the score to the explaination list
        suitable_ig_explaination_list.append(ig_grad_score[1])
        # add the full gradients list
        suitable_gradient_full_list.append(
            bucket.get_gradents_list_separated_by_class(ground_truth_lab)
        )

        # display the class only for the amount of times specified by the counter
        if show_element_counter < number_element_to_show:
            display_single_class(
                class_name=labels_name[ground_truth_lab],
                gradient=ig_grad_score[0],
                score=ig_grad_score[1],
                single_el=bucket.sample.detach().clone(),
                predicted=" ".join([labels_name[x] for x in pred]),
                arguments_folder=arguments_folder,
                idx=idx,
                prefix="confounded",
                correct=False,
            )
            show_element_counter += 1

        # if the score is equal to the maximal then we have that the 'suitable' is the maximum
        if ig_grad_score[1] == max_score_ig:
            suitable_is_max += 1
        else:
            suitable_is_not_max += 1

        tmp_max_score_image_list = 0
        # loop over the input gradient dictionary
        for int_label, ig_score_item in bucket.input_gradient_dict.items():
            str_groundtruth_label: str = labels_name[bucket.groundtruth_children]

            # add the max arguments
            if max_score_ig == ig_score_item[1]:
                tmp_max_score_image_list = ig_score_item[1]
                max_arguments_list.append(ig_score_item[1])

                # populate the max arguments dictionary
                if not str_groundtruth_label in max_arguments_dict:
                    max_arguments_dict[str_groundtruth_label] = {}
                if (
                    not labels_name[int_label]
                    in max_arguments_dict[str_groundtruth_label]
                ):
                    max_arguments_dict[str_groundtruth_label][
                        labels_name[int_label]
                    ] = 0
                max_arguments_dict[str_groundtruth_label][labels_name[int_label]] += 1

        # max label gradient
        max_val = -float("inf")
        for val in bucket.label_gradient.values():
            max_val = val[1] if val[1] > max_val else max_val
        max_ig_label_list_for_score_plot.append(([tmp_max_score_image_list.clone()], [max_val.clone()]))

        # comodo variables
        i_c_l: int = bucket.groundtruth_children
        i_p_l: int = bucket.groundtruth_parent
        s_c_l: str = labels_name[i_c_l]
        s_p_l: str = labels_name[i_p_l]
        label_args_dict: Dict[str, Dict[str, List[float]]] = dict()
        if not s_p_l in label_args_dict:
            label_args_dict[s_p_l] = {}
        # add the label gradient
        if not s_c_l in label_args_dict[s_p_l]:
            label_args_dict[s_p_l][s_c_l] = [
                bucket.label_gradient[i_c_l, i_p_l][1],
                1,
            ]
        label_args_dict[s_p_l][s_c_l][0] += bucket.label_gradient[i_c_l, i_p_l][1]
        label_args_dict[s_p_l][s_c_l][1] += 1

        # increase the influence depending on the result
        if bucket.label_gradient[i_c_l, i_p_l][1]:
            influence_parent_counter += 1
        else:
            not_influence_parent_counter += 1

        # increase the influence depending on the result
        if bucket.label_gradient[i_c_l, i_p_l][1]:
            influence_parent_counter += 1
        else:
            not_influence_parent_counter += 1

        # Analysis of the prediction TODO
        performed_predictions = (bucket.prediction == True).nonzero().flatten().tolist()
        has_influenced: bool = False
        for parent_pred in performed_predictions:
            for pred_el in performed_predictions:
                if (
                    pred_el,
                    parent_pred,
                ) in bucket.label_gradient and bucket.label_gradient[
                    pred_el, parent_pred
                ][
                    1
                ]:
                    has_influenced = True
                    break
            if has_influenced:
                break

        if has_influenced:
            prediction_influence_parent_counter += 1
        else:
            prediction_does_not_influence_parent_counter += 1
        #  print(idx, 'Going...')

    # return arguments
    return ArgumentsStepArgs(
        bucket_list=bucket_list,
        table_correlation=table_correlation,
        suitable_ig_explaination_list=suitable_ig_explaination_list,
        suitable_gradient_full_list=suitable_gradient_full_list,
        max_arguments_list=max_arguments_list,
        max_arguments_dict=max_arguments_dict,
        label_args_dict=label_args_dict,
        show_element_counter=show_element_counter,
        max_ig_label_list_for_score_plot=max_ig_label_list_for_score_plot,
        influence_parent_counter=influence_parent_counter,
        not_influence_parent_counter=not_influence_parent_counter,
        suitable_is_max=suitable_is_max,
        suitable_is_not_max=suitable_is_not_max,
        ig_lists=ig_lists,
        prediction_influence_parent_counter=prediction_influence_parent_counter,
        prediction_does_not_influence_parent_counter=prediction_does_not_influence_parent_counter,
        ig_lists_wrt_prediction=ig_lists_wrt_prediction,
        label_entropy_list=label_entropy_list,
        ig_entropy_list=ig_entropy_list
    )


def score_subclass_influence(
    sub_superclass_dict: Dict[str, Dict[str, List[int]]],
    title: str,
    folder: str,
) -> None:
    """Method which plots two bars plot for each superclass, showing which subclass has influenced the prediction
    Args:
        sub_superclass_dict [Dict[str, Dict[str, Tuple[int, int]]]]: dictionary which contains the name of the parent class as first key,
        the name of the superclass as second and the count associated with the influence as the final result
        title [str]: title of the plot
        folder [str]: folder name
    """
    for key in sub_superclass_dict.keys():
        fig = plt.figure(figsize=(8, 4))
        titles = np.array(list(sub_superclass_dict[key].keys()))
        values = np.array([el[1] for el in sub_superclass_dict[key].values()])
        plot = pd.Series(values).plot(kind="bar", color=["blue"])
        plot.bar_label(plot.containers[0], label_type="edge")
        plot.set_xticklabels(titles)
        plt.xticks(rotation=0)
        plt.title(
            "{} class {} #total {}".format(
                title, key, sum([el[0] for el in sub_superclass_dict[key].values()])
            )
        )
        plt.tight_layout()
        fig.savefig("{}/{}_{}.pdf".format(folder, key, title))
        plt.close(fig)


def plot_most_frequent_explainations(
    max_arg_dictionary: Dict[str, Dict[str, int]], title: str, folder: str
) -> None:
    """Method which plots two bars plot for each superclass, showing which class has the most frequent explainations
    Args:
        max_arg_dictionary [Dict[str, Dict[str, int]]]: dictionary which contains the name of the parent class as first key,
        the name of the superclass as second and the count associated with the maximum value as the final result
        title [str]: title of the plot
        folder [str]: folder name
    """
    for key in max_arg_dictionary.keys():
        fig = plt.figure(figsize=(8, 4))
        titles = np.array(list(max_arg_dictionary[key].keys()))
        values = np.array(list(max_arg_dictionary[key].values()))
        plot = pd.Series(values).plot(kind="bar", color=["blue"])
        plot.bar_label(plot.containers[0], label_type="edge")
        plot.set_xticklabels(titles)
        plt.xticks(rotation="vertical")
        plt.title(
            "{} class {} total #{}".format(
                title, key, sum(max_arg_dictionary[key].values())
            )
        )
        plt.tight_layout()
        fig.savefig("{}/{}_{}.pdf".format(folder, key, title))
        plt.close(fig)


def scatter_plot_score(
    ig_list_image: List[Tuple[List[float], List[float]]],
    conf_list_image: List[bool],
    correct_list_image: List[bool],
    ig_list_label: List[Tuple[List[float], List[float]]],
    correct_list_label: List[bool],
    ig_list_label_and_image: List[Tuple[List[float], List[float]]],
    correct_list_label_and_image: List[bool],
    folder: str,
    prefix: str,
) -> None:
    """Method which potrays a scatter plot showing the different input gradients score jittered
    and divided in six coloumns based on whether they come fro confounded/correct/data imabalnce samples
    Args:
        ig_list_image [List[Tuple[List[float], List[float]]]]: input gradients concerning image based confounders whether they are correctly or wrongly predicted
        conf_list_image [List[bool]]: list of boolean values depicting whether the examples are confounded or not
        correct_list_image [List[bool]]: list of boolean values depicting whether the examples are correctly predicted or not
        ig_list_label [List[Tuple[List[float], List[float]]]]: input gradientes concerning the label based confounders whether they are correctly or wrongly predicted
        correct_list_label [List[bool]]: list of boolean values depicting whether the examples are correctly predicted or not
        folder [str]: folder
        prefix [str]: prefix
    """

    corr_conf_counter = 0
    corr_not_conf_counter = 0
    not_corr_conf_counter = 0
    not_corr_not_conf_counter = 0
    corr_lab_counter = 0
    not_corr_lab_counter = 0
    corr_lab_image_counter = 0
    not_corr_lab_image_counter = 0

    # the zeros and ones counters
    corr_conf_0_1_c = [0, 0]
    corr_not_conf_0_1_c = [0, 0]
    not_corr_conf_0_1_c = [0, 0]
    not_corr_not_conf_0_1_c = [0, 0]
    corr_lab_0_1_c = [0, 0]
    not_corr_lab_0_1_c = [0, 0]
    corr_lab_image_0_1_c = [0, 0]
    not_corr_lab_image_0_1_c = [0, 0]

    score_list = list()
    label_list = list()

    for el_img, conf_img, corr_img in zip(
        ig_list_image,
        conf_list_image,
        correct_list_image,
    ):
        ig_grad_img, lab_grad_img = el_img
        is_correct = 0 if corr_img else 8
        is_conf = 0 if conf_img else 2
        for data in ig_grad_img:
            score_list.append([is_correct + is_conf + 0, data])
            label_list.append(int(corr_img))
        for data in lab_grad_img:
            score_list.append([is_correct + is_conf + 1, data])
            label_list.append(int(corr_img))

            corr_idx = 1 if data > 0 else 0

            # increasing the 0, 1 counters
            if corr_img and conf_img:
                corr_conf_0_1_c[corr_idx] += 1
            elif corr_img and not conf_img:
                corr_not_conf_0_1_c[corr_idx] += 1
            elif not corr_img and conf_img:
                not_corr_conf_0_1_c[corr_idx] += 1
            elif not corr_img and not conf_img:
                not_corr_not_conf_0_1_c[corr_idx] += 1

        # increasing the counters
        if corr_img and conf_img:
            corr_conf_counter += 1
        elif corr_img and not conf_img:
            corr_not_conf_counter += 1
        elif not corr_img and conf_img:
            not_corr_conf_counter += 1
        elif not corr_img and not conf_img:
            not_corr_not_conf_counter += 1

    for el_lab, corr_lab in zip(
        ig_list_label,
        correct_list_label,
    ):
        ig_grad_lab, lab_grad_lab = el_lab
        is_correct = 4 if corr_lab else 12
        for data in ig_grad_lab:
            score_list.append([is_correct + 0, data])
            label_list.append(int(corr_lab))
        for data in lab_grad_lab:
            score_list.append([is_correct + 1, data])
            label_list.append(int(corr_lab))

            corr_idx = 1 if data > 0 else 0
            if corr_lab:
                corr_lab_0_1_c[corr_idx] += 1
            elif not corr_lab:
                not_corr_lab_0_1_c[corr_idx] += 1

        if corr_lab:
            corr_lab_counter += 1
        elif not corr_lab:
            not_corr_lab_counter += 1

    # both confounded and label confounded
    for el_image_lab, corr_image_lab in zip(
        ig_list_label_and_image,
        correct_list_label_and_image,
    ):
        ig_grad_image_lab, lab_grad_image_lab = el_image_lab
        is_correct = 6 if corr_image_lab else 14
        for data in ig_grad_image_lab:
            score_list.append([is_correct + 0, data])
            label_list.append(int(corr_image_lab))
        for data in lab_grad_image_lab:
            score_list.append([is_correct + 1, data])
            label_list.append(int(corr_image_lab))

            corr_idx = 1 if data > 0 else 0
            if corr_image_lab:
                corr_lab_image_0_1_c[corr_idx] += 1
            elif not corr_image_lab:
                not_corr_lab_image_0_1_c[corr_idx] += 1

        if corr_image_lab:
            corr_lab_image_counter += 1
        elif not corr_image_lab:
            not_corr_lab_image_counter += 1

    label_list = np.array(label_list)
    score_list = np.array(score_list)

    # classificatore binario
    corr = scipy.stats.spearmanr(score_list[:, 1], label_list)

    # colors
    X, Y = score_list[:, 0], score_list[:, 1]

    colors = ["orange" if y == 1 else "blue" for y in label_list]

    fig = plt.figure(figsize=(15, 12))

    #  # jittering the X points
    #  for i in range(X.shape[0]):
    #      X[i] += np.random.uniform(low=-0.3, high=0.3)
    #
    #  for i in range(Y.shape[0]):
    #      if Y[i] == 0 or Y[i] == 1:
    #          Y[i] += np.random.uniform(low=-0.1, high=0.1)

    plt.scatter(X, Y, c=colors)
    plt.ylabel("gradient magnitude")
    plt.xlabel("type")

    plt.xticks(
        [0, 2, 4, 6, 8, 10, 12, 14],
        [
            f"Correct+Conf #{corr_conf_counter}",
            f"Correct+NotConf #{corr_not_conf_counter}",
            f"Correct+Imbalance #{corr_lab_counter}",
            f"Correct+Conf+Imbalance #{corr_lab_image_counter}",
            f"NotCorrect+Conf #{not_corr_conf_counter}",
            f"NotCorrect+NotConf #{not_corr_not_conf_counter}",
            f"NotCorrect+Imbalance #{not_corr_lab_counter}",
            f"NotCorrect+Conf+Imbalance #{not_corr_lab_image_counter}",
        ],
        rotation=20,
    )
    plt.title(
        "Spearman correlation {:.3f}, p-val {:.3f}\nsignificant with 95% confidence {} #conf {} #not-conf {}".format(
            corr[0],
            corr[1],
            corr[1] < 0.05,
            label_list.sum(),
            np.size(label_list) - label_list.sum(),
        ),
    )
    plt.legend()

    custom = [
        matplotlib.lines.Line2D(
            [], [], marker=".", markersize=20, color="orange", linestyle="None"
        ),
        matplotlib.lines.Line2D(
            [], [], marker=".", markersize=20, color="blue", linestyle="None"
        ),
    ]

    # add text
    plt.text(1 + 0.1, 0 + 0.01, str(corr_conf_0_1_c[0]), fontsize=9)
    plt.text(1 + 0.1, 0.8 + 0.01, str(corr_conf_0_1_c[1]), fontsize=9)

    plt.text(3 + 0.1, 0 + 0.01, str(corr_not_conf_0_1_c[0]), fontsize=9)
    plt.text(3 + 0.1, 0.8 + 0.01, str(corr_not_conf_0_1_c[1]), fontsize=9)

    plt.text(5 + 0.1, 0 + 0.01, str(corr_lab_0_1_c[0]), fontsize=9)
    plt.text(5 + 0.1, 0.8 + 0.01, str(corr_lab_0_1_c[1]), fontsize=9)

    plt.text(7 + 0.1, 0 + 0.01, str(corr_lab_image_0_1_c[0]), fontsize=9)
    plt.text(7 + 0.1, 0.8 + 0.01, str(corr_lab_image_0_1_c[1]), fontsize=9)

    plt.text(9 + 0.1, 0 + 0.01, str(not_corr_conf_0_1_c[0]), fontsize=9)
    plt.text(9 + 0.1, 0.8 + 0.01, str(not_corr_conf_0_1_c[1]), fontsize=9)

    plt.text(11 + 0.1, 0 + 0.01, str(not_corr_not_conf_0_1_c[0]), fontsize=9)
    plt.text(11 + 0.1, 0.8 + 0.01, str(not_corr_not_conf_0_1_c[1]), fontsize=9)

    plt.text(13 + 0.1, 0 + 0.01, str(not_corr_lab_0_1_c[0]), fontsize=9)
    plt.text(13 + 0.1, 0.8 + 0.01, str(not_corr_lab_0_1_c[1]), fontsize=9)

    plt.text(15 + 0.1, 0 + 0.01, str(not_corr_lab_image_0_1_c[0]), fontsize=9)
    plt.text(15 + 0.1, 0.8 + 0.01, str(not_corr_lab_image_0_1_c[1]), fontsize=9)

    plt.legend(custom, ["Correct (1)", "Not correct (2)"], fontsize=10)
    plt.close(fig)
    fig.savefig(
        "{}/{}_full_gradients_correlations.pdf".format(
            folder,
            prefix,
        )
    )


def box_plot_input_gradients(
    ig_list_image_corr_conf: List[float],
    ig_list_image_corr_not_conf: List[float],
    ig_list_image_not_corr_conf: List[float],
    ig_list_image_not_corr_not_conf: List[float],
    ig_list_label_not_corr: List[float],
    ig_list_label_corr: List[float],
    ig_list_lab_image_not_corr_conf: List[float],
    ig_list_lab_image_corr_conf: List[float],
    ig_list_image_corr_conf_counter: int,
    ig_list_image_corr_not_conf_counter: int,
    ig_list_image_not_corr_conf_counter: int,
    ig_list_image_not_corr_not_conf_counter: int,
    ig_list_label_not_corr_counter: int,
    ig_list_label_corr_counter: int,
    ig_list_lab_image_not_corr_conf_counter: int,
    ig_list_lab_image_corr_conf_counter: int,
    folder: str,
    prefix: str,
) -> None:
    """Procedure which portrays the input gradient box plot for each of the analyzed cases
    Args:
        ig_list_image_corr_conf [List[float]]: input gradient list concerning confounded samples who have been correctly guessed
        ig_list_image_corr_not_conf [List[float]]: input gradient list concerning not confounded samples who have been correctly guessed
        ig_list_image_not_corr_conf [List[float]]: input gradient list concerning confounded samples who have been wrongly guessed
        ig_list_image_not_corr_not_conf [List[float]]: input gradient list concerning not confounded samples who have been wrongly guessed
        ig_list_label_not_corr [List[float]]: input gradient list concerning label imbalance confounded samples who have been correctly guessed
        ig_list_label_corr [List[float]]: input gradient list concerning label imbalance confounded samples who have been wrongly guessed
        ig_list_image_corr_conf_counter [int]: number of input gradient list concerning confounded samples who have been correctly guessed
        ig_list_image_corr_not_conf_counter [int]: number of input gradient list concerning not confounded samples who have been correctly guessed
        ig_list_image_not_corr_conf_counter [int]: number of input gradient list concerning not confounded samples who have been wrongly guessed
        ig_list_image_not_corr_not_conf_counter [int]: number of input gradient list concerning label imbalance confounded samples who have been correctly guessed
        ig_list_label_not_corr_counter [int]: number of input gradient list concerning label imbalance confounded samples who have been wrongly guessed
        ig_list_label_corr_counter [int]: number of number of input gradient list concerning confounded samples who have been correctly guessed
        folder [str]: folder name
        prefix [str]: prefix for the prediction
    """

    fig = plt.figure(figsize=(15, 12))
    bp_dict = plt.boxplot(
        [
            ig_list_image_corr_conf,
            ig_list_image_corr_not_conf,
            ig_list_image_not_corr_conf,
            ig_list_image_not_corr_not_conf,
            ig_list_label_not_corr,
            ig_list_label_corr,
            ig_list_lab_image_not_corr_conf,
            ig_list_lab_image_corr_conf,
        ]
    )
    plt.xticks(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [
            f"Correct+Conf #{ig_list_image_corr_conf_counter}",
            f"Correct+NotConf #{ig_list_image_corr_not_conf_counter}",
            f"NotCorrect+Conf #{ig_list_image_not_corr_conf_counter}",
            f"NotCorrect+NotConf #{ig_list_image_not_corr_not_conf_counter}",
            f"Correct+Imbalance #{ig_list_label_not_corr_counter}",
            f"NotCorrect+Imbalance #{ig_list_label_corr_counter}",
            f"NotCorrect+Conf+Imbalance #{ig_list_lab_image_not_corr_conf_counter}",
            f"Correct+Conf+Imbalance #{ig_list_lab_image_corr_conf_counter}",
        ],
        rotation=20,
    )

    for line in bp_dict["medians"]:
        # get position data for median line
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        plt.text(x, y, "%.5f" % y, horizontalalignment="center")

    fig.suptitle("Boxplot all categories")

    fig.savefig(
        "{}/{}_input_gradient_boxplot.pdf".format(
            folder,
            prefix,
        )
    )

    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    bp_dict = plt.boxplot(
        [
            list(
                itertools.chain(
                    ig_list_image_corr_conf,
                    ig_list_image_corr_not_conf,
                    ig_list_label_corr,
                    ig_list_lab_image_corr_conf,
                )
            ),
            list(
                itertools.chain(
                    ig_list_image_corr_not_conf,
                    ig_list_image_not_corr_conf,
                    ig_list_label_not_corr,
                    ig_list_lab_image_not_corr_conf,
                )
            ),
        ]
    )

    plt.xticks(
        [1, 2],
        [
            f"Correct #{ig_list_image_corr_conf_counter + ig_list_image_corr_not_conf_counter + ig_list_label_corr_counter + ig_list_lab_image_corr_conf_counter}",
            f"NotCorrect #{ig_list_image_not_corr_conf_counter + ig_list_image_not_corr_not_conf_counter + ig_list_label_corr_counter + ig_list_lab_image_not_corr_conf_counter}",
        ],
    )

    for line in bp_dict["medians"]:
        # get position data for median line
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        plt.text(x, y, "%.5f" % y, horizontalalignment="center")

    fig.suptitle("Boxplot correct and not correct")

    fig.savefig(
        "{}/{}_input_gradient_correct_not_correct_boxplot.pdf".format(
            folder,
            prefix,
        )
    )

    plt.close(fig)


def plot_correlation_table(
    ig_list: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]
) -> None:
    """Plots the correlation table given the integrated gradient list
    Args:
        ig_list: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]: integrated gradient list
    """
    for class_ in ig_list.keys():
        corr_text_dict = {}
        for lab in ig_list[class_].keys():
            corr_text_dict[lab] = scipy.stats.spearmanr(
                np.array([el[1] for el in ig_list[class_][lab]]),
                np.array([el[0] for el in ig_list[class_][lab]]),
            )
            if np.isnan(corr_text_dict[lab].correlation):
                corr_text_dict[lab] = 0

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")

        df = pd.DataFrame(
            [list(corr_text_dict.values())], columns=list(corr_text_dict.keys())
        )
        ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        fig.tight_layout()


def get_samples_for_arguments(
    net: nn.Module,
    dataset: str,
    dataloaders: Dict[str, Any],
    device: str,
    force_prediction: bool,
    use_softmax: bool,
    prediction_treshold: float,
    loader: torch.utils.data.DataLoader,
    correct: bool,
    num_element_to_analyze: int,
    test: dotdict,
    confounded: bool,
    phase: str,
    use_probabilistic_circuits: bool = False,
    cmpe: CircuitMPE = None,
    gate: DenseGatingFunction = None,
) -> List[Tuple[torch.Tensor, List[int], List[int]]]:
    """
    Retrieve some samples from the dataloaders for a specific class according to whether
    one wants the example to be confounded and/or correcly classified

    Args:
        net [nn.Module]: neural network
        dataset [str]: dataset name
        dataloaders [Dict[str, Any]]: dataloaders
        device [str]: device name
        force_prediction [bool]: whether to force the prediction
        use_softmax [bool]: whether to use softmax
        prediction_treshold [float]: prediction threshold
        loader [torch.utils.data.DataLoader]: dataloaders
        correct [bool]: whether the user wants the example to be correctly classified
        num_element_to_analyze [int]: number of elements to analyze
        test [dotdict]: test dict
        confounded [bool] whether the sample should be confounded
        phase [str]: which phase we are interested to
    Returns:
        List[Tuple[torch.Tensor, List[int], List[int]]]: list of sample, predicted classes list and groundtruth label examples
    """

    # datalist
    data_list: List[Tuple[torch.Tensor, List[int], List[int]]] = list()
    # whether the process is completed
    completed = False
    # get confounders
    confounders = get_confounders(dataset)
    # already added tensors
    already_added_tensors = list()

    print("Len", len(loader))
    # loop over the loader
    for _, inputs in tqdm.tqdm(
        enumerate(loader), desc="Return samples from dataloader"
    ):
        # Unpack data
        (inputs, superclass, subclass, targets) = inputs

        # loop over the samples in the batch
        for el_idx in range(inputs.shape[0]):
            confounded_sample = False
            found = False
            if superclass[el_idx] in confounders:
                for item in confounders[superclass[el_idx]][phase]:
                    if "subclass" in item and item["subclass"] == subclass[el_idx]:
                        confounded_sample = True
                    if found:
                        break
            # if we are ninterested in confounded samples and it is confounded then we are ok
            # same with not confounded samples and we are not interested with confounded samples
            if (confounded and not confounded_sample) or (
                not confounded and confounded_sample
            ):
                continue
            # single target
            single_target = torch.unsqueeze(targets[el_idx], 0)
            single_target_bool = single_target > 0.5
            # single el
            single_el = torch.unsqueeze(inputs[el_idx], 0)
            single_el.requires_grad = True
            # move to device
            single_el = single_el.to(device)
            # forward pass
            logits = net(single_el.float())
            # predicted value
            if use_probabilistic_circuits:
                # thetas
                thetas = gate(logits.float())
                # negative log likelihood and map
                cmpe.set_params(thetas)
                predicted_1_0 = (cmpe.get_mpe_inst(single_el.shape[0]) > 0).long()
            elif force_prediction:
                predicted_1_0 = force_prediction_from_batch(
                    logits.data.cpu(),
                    prediction_treshold,
                    use_softmax,
                    dataloaders["train_set"].n_superclasses,
                )
            else:
                predicted_1_0 = logits.data.cpu() > prediction_treshold  # 0.5

            # to float
            predicted_bool = predicted_1_0
            predicted_1_0 = predicted_1_0.to(torch.float)[0]

            # if we are interested in correctly predicting sample or we are not interested in doing that: add it to the list
            if (
                correct
                and torch.equal(
                    predicted_bool[:, test.to_eval], single_target_bool[:, test.to_eval]
                )
            ) or (
                not correct
                and not torch.equal(
                    predicted_bool[:, test.to_eval], single_target_bool[:, test.to_eval]
                )
            ):
                #  if correct:
                #  print(subclass[el_idx])
                    #  if any(torch.equal(inputs[el_idx], tensor) for tensor in already_added_tensors):
                    #      print("Tensor is present in the list.")
                    #  else:
                    #      print("Tensor is not present in the list.")

                if correct and confounded:
                    gradient_to_show = torch.autograd.grad(
                        outputs=logits,
                        inputs=single_el,
                        grad_outputs=torch.ones_like(logits),
                        retain_graph=True,
                    )[0]

                    gradient_to_show = torch.squeeze(gradient_to_show, 0)
                    gradient_to_show = np.fabs(gradient_to_show.detach().numpy().transpose(1, 2, 0))
                    single_el = torch.squeeze(single_el, 0)
                    single_el = single_el.detach().numpy().transpose(1, 2, 0)

                    # norm color
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(gradient_to_show))
                    # show the picture
                    fig = plt.figure()
                    plt.colorbar(
                        matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
                        label="Gradient magnitude",
                    )
                    plt.imshow(single_el, cmap="gray")
                    plt.imshow(gradient_to_show, cmap="viridis", alpha=0.5)
                    plt.subplots_adjust(top=0.72)
                    plt.title("Input gradient correctly predicted and watermarked")

                    # show the figure
                    fig.savefig(
                        "lol/correct_sample_watermarked.pdf",
                        dpi=fig.dpi,
                    )
                    plt.close(fig)

                data_list.append(
                    (
                        inputs[el_idx],
                        predicted_bool[:, test.to_eval]
                        .squeeze(0)
                        .nonzero()
                        .squeeze(-1)
                        .tolist(),
                        single_target_bool[:, test.to_eval]
                        .squeeze(0)
                        .nonzero()
                        .squeeze(-1)
                        .tolist(),
                    )
                )

                already_added_tensors.append(inputs[el_idx])
            # add element to the datalist
            if len(data_list) >= num_element_to_analyze:
                completed = True
                break

            #  # TODO remove
            #  if len(data_list) == 3:
            #      completed = True
            #      break

        # completed
        if completed:
            break
    # return the datalist
    return data_list


def main(args: Namespace) -> None:
    """Checks the command line arguments and then runs the debug

    Args:
        args (Namespace): command line arguments
    """
    print("\n### Network arguments ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

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
            dataloaders["train_R"],
            output_classes,
            args.constrained_layer,
            channels=img_depth,
            img_height=img_size,
            img_width=img_size,
        )  # 20 superclasses, 100 subclasses + the root
    else:
        net = ResNet18(
            dataloaders["train_R"], 121, args.constrained_layer
        )  # 20 superclasses, 100 subclasses + the root

    # move everything on the cpu
    net = net.to(args.device)
    net.R = net.R.to(args.device)
    net.eval()

    # use the probabilistic circuit
    cmpe: CircuitMPE = None
    gate: DenseGatingFunction = None

    if args.use_probabilistic_circuits:
        print("Using probabilistic circuits...")
        cmpe, gate = prepare_probabilistic_circuit(
            dataloaders["train_set"].get_A(),
            args.constraint_folder,
            args.dataset,
            args.device,
            args.gates,
            args.num_reps,
            output_classes,
            args.S,
        )

    # summary
    summary(net, (img_depth, img_size, img_size))

    # Test on best weights (of the confounded model)
    load_last_weights(net, args.weights_path_folder, args.device)
    #load_best_weights_gate(gate, args.weights_path_folder, args.device)

    if args.use_probabilistic_circuits:
        gate.eval()

    print("Network resumed...")
    print("-----------------------------------------------------")

    print("#> Arguments...")
    arguments(net=net, dataloaders=dataloaders, cmpe=cmpe, gate=gate, **vars(args))
