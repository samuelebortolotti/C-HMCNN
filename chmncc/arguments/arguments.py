"""Analyze arguments"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
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
    dotdict,
    split,
    get_confounders,
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
    """Max arguments dictionary """
    max_arguments_dict: Dict[str, Dict[str, int]]
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the label gradient]"""
    label_args_dict: Dict[str, Dict[str, List[int]]]
    """Counter for showing the single element example"""
    show_element_counter: int
    """Counter which defines how many times a subclass has influenced a superclass"""
    influence_parent_counter: int
    """Counter which defines how many times a subclass has not influenced a superclass"""
    not_influence_parent_counter: int
    """Counter which defined how many times the 'right' argument is the maximuum one"""
    suitable_is_max: int
    """Counter which defined how many times the 'right' argument is not the maximuum one"""
    suitable_is_not_max: int

    def __init__(
        self,
        bucket_list: List[ArgumentBucket],
        table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]],
        suitable_ig_explaination_list: List[float],
        suitable_gradient_full_list: List[Tuple[List[float], List[float]]],
        ig_lists: List[Tuple[List[float], List[float]]],
        max_arguments_list: List[float],
        max_arguments_dict: Dict[str, Dict[str, int]],
        label_args_dict: Dict[str, Dict[str, List[int]]],
        show_element_counter: int,
        influence_parent_counter: int,
        not_influence_parent_counter: int,
        suitable_is_max: int,
        suitable_is_not_max: int,
    ):
        self.bucket_list = bucket_list
        self.table_correlation = table_correlation
        self.suitable_ig_explaination_list = suitable_ig_explaination_list
        self.suitable_gradient_full_list = suitable_gradient_full_list
        self.ig_lists = ig_lists
        self.max_arguments_list = max_arguments_list
        self.max_arguments_dict = max_arguments_dict
        self.label_args_dict = label_args_dict
        self.show_element_counter = show_element_counter
        self.influence_parent_counter = influence_parent_counter
        self.not_influence_parent_counter = not_influence_parent_counter
        self.suitable_is_max = suitable_is_max
        self.suitable_is_not_max = suitable_is_not_max


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
        default=2
    )
    parser.add_argument(
        "--tau",
        "-t",
        dest="tau",
        type=float,
        help="Tau for gradient analysis table",
        default=0.5
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
        fig.savefig("{}/gradients_{}_{}.png".format(arguments_folder, str(idx), i))
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
    fig = plt.figure(figsize=(8, 4))
    titles = np.array([label_1, label_2])
    values = np.array([x1, x2])
    plot = pd.Series(values).plot(kind="bar", color=["green", "red"])
    plot.bar_label(plot.containers[0], label_type="edge")
    plot.set_xticklabels(titles)
    plt.xticks(rotation=0)
    plt.title("{}".format(title))
    plt.tight_layout()
    fig.savefig("{}/{}.png".format(folder, title))
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
    plot = pd.Series(values).plot(kind="bar", color=color)
    plot.bar_label(plot.containers[0], label_type="edge")
    plot.set_xticklabels(titles)
    plt.xticks(rotation="vertical")
    plt.title("{}".format(title))
    plt.tight_layout()
    fig.savefig("{}/{}.png".format(folder, title))
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
    # TODO nome forviante
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
        "{}/{}_integrated_gradient_correlation.png".format(
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
        "{}/{}_integrated_gradient_boxplot.png".format(
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
        "{}/{}_{}_input_gradient_{}_{}.png".format(
            arguments_folder,
            prefix,
            class_name,
            idx,
            "_correct" if correct else "",
        ),
        dpi=fig.dpi,
    )
    print(
        "Saving {}/{}_{}_input_gradient_{}_{}.png".format(
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
        )
        print("Corr+Conf #", len(correct_confound.bucket_list))

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
        )
        print("Wrong+Conf #", len(wrong_confound.bucket_list))

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
        )
        print("Corr+NotConf #", len(correct_not_confound.bucket_list))

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
        )
        print("NotCorr+NotConf #", len(wrong_not_confound.bucket_list))

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
        )
        print("Corr+Imbalance #", len(correct_lab.bucket_list))

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
        )
        print("NotCorr+Imbalance #", len(wrong_lab.bucket_list))

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
        )
        print("NotCorr+Conf+Imbalance #", len(wrong_lab_img.bucket_list))

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
        )
        print("Corr+Conf+Imbalance #", len(corr_lab_img.bucket_list))

        plot_arguments(
            correct_confound,
            wrong_confound,
            correct_not_confound,
            wrong_not_confound,
            correct_lab,
            wrong_lab,
            wrong_lab_img,
            corr_lab_img,
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


def plot_arguments(
    correct_confound: ArgumentsStepArgs,
    wrong_confound: ArgumentsStepArgs,
    correct_not_confound: ArgumentsStepArgs,
    wrong_not_confound: ArgumentsStepArgs,
    correct_lab: ArgumentsStepArgs,
    wrong_lab: ArgumentsStepArgs,
    wrong_lab_img_confound: ArgumentsStepArgs,
    correct_lab_img_confound: ArgumentsStepArgs,
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
        ig_list_lab_image_not_corr_conf=[el for l in wrong_lab_img_confound.ig_lists for el in l[0]],
        ig_list_lab_image_corr_conf=[el for l in correct_lab_img_confound.ig_lists for el in l[0]],
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
        ig_list_lab_image_not_corr_conf=[el for l in wrong_lab_img_confound.ig_lists for el in l[1]],
        ig_list_lab_image_corr_conf=[el for l in correct_lab_img_confound.ig_lists for el in l[1]],
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
        ig_list_lab_image_not_corr_conf=[el for l in wrong_lab_img_confound.ig_lists for e in l for el in e],
        ig_list_lab_image_corr_conf=[el for l in correct_lab_img_confound.ig_lists for e in l for el in e],
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
    input_gradient_scatter(
        correct_guesses=correct_confound.max_arguments_list,
        correct_guesses_conf=correct_guesses_conf,
        wrongly_guesses=wrong_confound.max_arguments_list,
        wrongly_guesses_conf=wrongly_guesses_conf,
        folder=arguments_folder,
        prefix="max_score",
    )

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

    max_arg_dictionary = correct_confound.max_arguments_dict

    plot_most_frequent_explainations(
        correct_confound.max_arguments_dict, "Max chosen per class: [Correct+Conf]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, wrong_confound.max_arguments_dict
    )

    plot_most_frequent_explainations(
        wrong_confound.max_arguments_dict, "Max chosen per class: [NotCorrect+Conf]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, correct_not_confound.max_arguments_dict
    )

    plot_most_frequent_explainations(
        correct_not_confound.max_arguments_dict, "Max chosen per class: [Correct+NotConf]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, wrong_not_confound.max_arguments_dict
    )

    plot_most_frequent_explainations(
        wrong_not_confound.max_arguments_dict, "Max chosen per class: [NotCorrect+NotConf]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, correct_lab.max_arguments_dict
    )

    plot_most_frequent_explainations(
        correct_lab.max_arguments_dict, "Max chosen per class: [Correct+Imbalance]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, wrong_lab.max_arguments_dict
    )

    plot_most_frequent_explainations(
        wrong_lab.max_arguments_dict, "Max chosen per class: [NotCorrect+Imbalance]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, wrong_lab_img_confound.max_arguments_dict
    )

    plot_most_frequent_explainations(
        wrong_lab_img_confound.max_arguments_dict, "Max chosen per class: [NotCorrect+Conf+Imbalance]", arguments_folder
    )

    max_arg_dictionary = sum_merge_dictionary(
        max_arg_dictionary, correct_lab_img_confound.max_arguments_dict
    )

    plot_most_frequent_explainations(
        correct_lab_img_confound.max_arguments_dict, "Max chosen per class: [Correct+Conf+Imbalance]", arguments_folder
    )

    plot_most_frequent_explainations(
        max_arg_dictionary, "Max chosen per class", arguments_folder
    )

    plot_gradient_analysis_table_max(
        wrong_lab=wrong_lab.bucket_list,
        wrong_confound=wrong_confound.bucket_list,
        wrong_lab_img_confound=wrong_lab_img_confound.bucket_list,
        wrong_ok=wrong_not_confound.bucket_list,
        arguments_folder=arguments_folder,
        tau=tau
    )

    plot_gradient_analysis_table_full(
        wrong_lab=wrong_lab.bucket_list,
        wrong_confound=wrong_confound.bucket_list,
        wrong_lab_img_confound=wrong_lab_img_confound.bucket_list,
        wrong_ok=wrong_not_confound.bucket_list,
        arguments_folder=arguments_folder,
        tau=tau
    )


def plot_gradient_analysis_table_max(
    wrong_lab: List[ArgumentBucket],
    wrong_confound: List[ArgumentBucket],
    wrong_lab_img_confound: List[ArgumentBucket],
    wrong_ok: List[ArgumentBucket],
    arguments_folder: str,
    tau: float
):

    # TODO considerando valore massimo e considerando tutti

    columns_header = ['# X > t', '# Y > t', '# X > # Y', '#']
    rows_header = ['Not correct: confound on X', 'Not correct: confound on Y', 'Not correct: confound on both XY', 'Not correct: no confound']

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
        ["lightgreen","w","lightgreen","w"],
        ["w", "lightgreen","lightcoral","w"],
        ["w", "w", "lightcoral","w"],
        ["w", "w","w","w"],
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

    rcolors = np.full(len(rows_header), 'linen')
    ccolors = np.full(len(columns_header), 'lavender')

    table = ax1.table(
        cellText=data,
        cellLoc='center',
        rowLabels=rows_header,
        rowColours=rcolors,
        rowLoc='center',
        colColours=ccolors,
        cellColours=colors,
        colLabels=columns_header,
        loc='center'
    )

    table.scale(1, 2)
    table.set_fontsize(16)
    ax1.axis('off')
    title = f"Gradient Analysis (per samples) Table: tau = {tau}"
    ax1.set_title(f'{title}', weight='bold', size=14, color='k')
    fig.savefig("{}/{}.png".format(arguments_folder, title), bbox_inches='tight')
    plt.close(fig)


def plot_gradient_analysis_table_full(
    wrong_lab: List[ArgumentBucket],
    wrong_confound: List[ArgumentBucket],
    wrong_lab_img_confound: List[ArgumentBucket],
    wrong_ok: List[ArgumentBucket],
    arguments_folder: str,
    tau: float
):

    # TODO considerando valore massimo e considerando tutti

    columns_header = ['# X > t', '# Y > t', '# X > # Y', '#ig', '#labg']
    rows_header = ['Not correct: confound on X', 'Not correct: confound on Y', 'Not correct: confound on both XY', 'Not correct: no confound']

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
        ["lightgreen","w","lightgreen","w", "w"],
        ["w", "lightgreen","lightcoral","w", "w"],
        ["w", "w", "lightcoral","w", "w"],
        ["w", "w","w","w", "w"],
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

    rcolors = np.full(len(rows_header), 'linen')
    ccolors = np.full(len(columns_header), 'lavender')

    table = ax1.table(
        cellText=data,
        cellLoc='center',
        rowLabels=rows_header,
        rowColours=rcolors,
        rowLoc='center',
        colColours=ccolors,
        cellColours=colors,
        colLabels=columns_header,
        loc='center'
    )

    table.scale(1, 2)
    table.set_fontsize(16)
    ax1.axis('off')
    title = f"Gradient Analysis (all gradients) Table: tau = {tau}"
    ax1.set_title(f'{title}', weight='bold', size=14, color='k')
    fig.savefig("{}/{}.png".format(arguments_folder, title), bbox_inches='tight')
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
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the input gradient]"""
    max_arguments_dict: Dict[str, Dict[str, int]] = dict()
    """Dictionary of arguments of the highest scoring subclass per each superclass [concerning the label gradient]"""
    label_args_dict: Dict[str, Dict[str, List[int]]] = dict()
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
        )

        # ig lists
        ig_lists.append(bucket.get_gradents_list_separated())
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
            # no need to show the barplot list for each explaination
            #  ig_list, ig_titles = bucket.get_gradients_list_and_names()
            #  single_element_barplot(
            #      labels_name[ground_truth_lab], idx, ig_list, ig_titles, arguments_folder
            #  )
            show_element_counter += 1

        # if the score is equal to the maximal then we have that the 'suitable' is the maximum
        if ig_grad_score[1] == max_score_ig:
            suitable_is_max += 1
        else:
            suitable_is_not_max += 1

        # loop over the input gradient dictionary
        for int_label, ig_score_item in bucket.input_gradient_dict.items():
            str_groundtruth_label: str = labels_name[bucket.groundtruth_children]

            # add the max arguments
            if max_score_ig == ig_score_item[1]:
                max_arguments_list.append(ig_score_item[1])

                # populate the max arguments dictionary
                if not str_groundtruth_label in max_arguments_dict:
                    max_arguments_dict[str_groundtruth_label] = {}
                if not labels_name[int_label] in max_arguments_dict[str_groundtruth_label]:
                    max_arguments_dict[str_groundtruth_label][labels_name[int_label]] = 0
                max_arguments_dict[str_groundtruth_label][labels_name[int_label]] += 1

        # comodo variables
        i_c_l: int = bucket.groundtruth_children
        i_p_l: int = bucket.groundtruth_parent
        s_c_l: str = labels_name[i_c_l]
        s_p_l: str = labels_name[i_p_l]
        label_args_dict: Dict[str, Dict[str, List[int]]] = dict()
        if not s_p_l in label_args_dict:
            label_args_dict[s_p_l] = {}
        # add the label gradient
        if not s_c_l in label_args_dict[s_p_l]:
            label_args_dict[s_p_l][s_c_l] = [
                int(bucket.label_gradient[i_c_l, i_p_l][1]),
                1,
            ]
        label_args_dict[s_p_l][s_c_l][0] += int(bucket.label_gradient[i_c_l, i_p_l][1])
        label_args_dict[s_p_l][s_c_l][1] += 1

        # increase the influence depending on the result
        if bucket.label_gradient[i_c_l, i_p_l][1]:
            influence_parent_counter += 1
        else:
            not_influence_parent_counter += 1

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
        influence_parent_counter=influence_parent_counter,
        not_influence_parent_counter=not_influence_parent_counter,
        suitable_is_max=suitable_is_max,
        suitable_is_not_max=suitable_is_not_max,
        ig_lists=ig_lists,
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
        fig.savefig("{}/{}_{}.png".format(folder, key, title))
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
        fig.savefig("{}/{}_{}.png".format(folder, key, title))
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
            score_list.append([is_correct + is_conf + 1, int(data)])
            label_list.append(int(corr_img))

            # increasing the 0, 1 counters
            if corr_img and conf_img:
                corr_conf_0_1_c[int(data)] += 1
            elif corr_img and not conf_img:
                corr_not_conf_0_1_c[int(data)] += 1
            elif not corr_img and conf_img:
                not_corr_conf_0_1_c[int(data)] += 1
            elif not corr_img and not conf_img:
                not_corr_not_conf_0_1_c[int(data)] += 1


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
            score_list.append([is_correct + 1, int(data)])
            label_list.append(int(corr_lab))

            if corr_lab:
                corr_lab_0_1_c[int(data)] += 1
            elif not corr_lab:
                not_corr_lab_0_1_c[int(data)] += 1

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
            score_list.append([is_correct + 1, int(data)])
            label_list.append(int(corr_image_lab))

            if corr_image_lab:
                corr_lab_image_0_1_c[int(data)] += 1
            elif not corr_image_lab:
                not_corr_lab_image_0_1_c[int(data)] += 1

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

    # jittering the X points
    for i in range(X.shape[0]):
        X[i] += np.random.uniform(low=-0.3, high=0.3)

    for i in range(Y.shape[0]):
        if Y[i] == 0 or Y[i] == 1:
            Y[i] += np.random.uniform(low=-0.1, high=0.1)

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
    plt.text(1, 0 + 0.01, str(corr_conf_0_1_c[0]), fontsize=9)
    plt.text(1, 1 + 0.01, str(corr_conf_0_1_c[1]), fontsize=9)

    plt.text(3, 0 + 0.01, str(corr_not_conf_0_1_c[0]), fontsize=9)
    plt.text(3, 1 + 0.01, str(corr_not_conf_0_1_c[1]), fontsize=9)

    plt.text(5, 0 + 0.01, str(corr_lab_0_1_c[0]), fontsize=9)
    plt.text(5, 1 + 0.01, str(corr_lab_0_1_c[1]), fontsize=9)

    plt.text(7, 0 + 0.01, str(corr_lab_image_0_1_c[0]), fontsize=9)
    plt.text(7, 1 + 0.01, str(corr_lab_image_0_1_c[1]), fontsize=9)

    plt.text(9, 0 + 0.01, str(not_corr_conf_0_1_c[0]), fontsize=9)
    plt.text(9, 1 + 0.01, str(not_corr_conf_0_1_c[1]), fontsize=9)

    plt.text(11, 0 + 0.01, str(not_corr_not_conf_0_1_c[0]), fontsize=9)
    plt.text(11, 1 + 0.01, str(not_corr_not_conf_0_1_c[1]), fontsize=9)

    plt.text(13, 0 + 0.01, str(not_corr_lab_0_1_c[0]), fontsize=9)
    plt.text(13, 1 + 0.01, str(not_corr_lab_0_1_c[1]), fontsize=9)

    plt.text(15, 0 + 0.01, str(not_corr_lab_image_0_1_c[0]), fontsize=9)
    plt.text(15, 1 + 0.01, str(not_corr_lab_image_0_1_c[1]), fontsize=9)

    plt.legend(custom, ["Correct (1)", "Not correct (2)"], fontsize=10)
    plt.close(fig)
    fig.savefig(
        "{}/{}_full_gradients_correlations.png".format(
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
            ig_list_lab_image_corr_conf
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
        "{}/{}_input_gradient_boxplot.png".format(
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
                    ig_list_lab_image_corr_conf
                )
            ),
            list(
                itertools.chain(
                    ig_list_image_corr_not_conf,
                    ig_list_image_not_corr_conf,
                    ig_list_label_not_corr,
                    ig_list_lab_image_not_corr_conf
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
        "{}/{}_input_gradient_correct_not_correct_boxplot.png".format(
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
            if force_prediction:
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
            # add element to the datalist
            if len(data_list) >= num_element_to_analyze:
                completed = True
                break
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
    # summary
    summary(net, (img_depth, img_size, img_size))

    # Test on best weights (of the confounded model)
    load_best_weights(net, args.weights_path_folder, args.device)

    print("Network resumed...")
    print("-----------------------------------------------------")

    print("#> Arguments...")
    arguments(net=net, dataloaders=dataloaders, **vars(args))
