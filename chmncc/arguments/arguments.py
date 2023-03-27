"""Analyze arguments"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
import os
import torch.nn as nn
import scipy
import itertools
import pandas as pd
from chmncc.networks import ResNet18, LeNet5, LeNet7, AlexNet, MLP
from chmncc.config import (
    cifar_confunders,
    mnist_confunders,
    fashion_confunders,
    omniglot_confunders,
)
from chmncc.utils.utils import (
    force_prediction_from_batch,
    load_best_weights,
    dotdict,
    split,
)
from chmncc.dataset import (
    load_dataloaders,
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
    bucket_list: List[ArgumentBucket]
    table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]
    suitable_ig_explaination_list: List[float]
    suitable_gradient_full_list: List[Tuple[List[float], List[float]]]
    max_arguments_list: List[float]
    ig_lists: List[Tuple[List[float], List[float]]]
    max_arguments_dict: Dict[str, Dict[str, int]]
    label_args_dict: Dict[str, Dict[str, List[int]]]
    show_element_counter: int
    influence_parent_counter: int
    not_influence_parent_counter: int
    suitable_is_max: int
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
        suitable_is_not_max: int
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
        plt.close()


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
    plt.close()


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
    plt.close()


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
    plt.close()


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
    **kwargs: Any
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
        **kwargs [Any]: kwargs
    """

    # labels name
    labels_name = dataloaders["test_set"].nodes_names_without_root

    print("Have to run for {} arguments iterations...".format(iterations))

    # running for the requested iterations
    for it in range(iterations):

        print("Start iteration number {}".format(it))
        print("-----------------------------------------------------")

        correct_arguments_args = arguments_step(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            correct_samples_only=True,
            confounded_samples_only=True,
            num_element_to_analyze=3,
            labels_name=labels_name,
            number_element_to_show=1,
            arguments_folder=arguments_folder
        )

        wrong_arguments_args = arguments_step(
            net = net,
            dataset = dataset,
            dataloaders = dataloaders,
            device = device,
            force_prediction = force_prediction,
            use_softmax = use_softmax,
            prediction_treshold = prediction_treshold,
            correct_samples_only = True,
            confounded_samples_only = True,
            num_element_to_analyze = 3,
            labels_name = labels_name,
            number_element_to_show = 1,
            arguments_folder = arguments_folder
        )

        plot_arguments(correct_arguments_args, wrong_arguments_args, arguments_folder=arguments_folder)


def get_table_correlation_dictionary_from(
    bucket: ArgumentBucket,
    correct_samples_only: bool,
    table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]
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
        table_correlation[bucket.groundtruth_children][key].append((correct_samples_only, value))
    return table_correlation


def plot_arguments(correct: ArgumentsStepArgs, wrong: ArgumentsStepArgs, arguments_folder: str) -> None:
    """Produce all the plots for the arguments
    Args:
        correct [ArgumentsStepArgs]: arguments step args for the correct data
        wrong [ArgumentsStepArgs]: arguments step args for the wrong data
        arguments_folder [str]: arguments folder
    """
    correct_guesses = [
        el[1]
        for bucket in correct.bucket_list
        for el in bucket.input_gradient_dict.values()
    ]
    correct_guesses_confounded = [True for _ in range(len(correct_guesses))]
    wrongly_guesses = [
        el[1]
        for bucket in wrong.bucket_list
        for el in bucket.input_gradient_dict.values()
    ]
    wrongly_guesses_confounded = [False for _ in range(len(wrongly_guesses))]
    input_gradient_scatter(
        correct_guesses=correct_guesses,
        correct_guesses_conf=correct_guesses_confounded,
        wrongly_guesses=wrongly_guesses,
        wrongly_guesses_conf=wrongly_guesses_confounded,
        folder=arguments_folder,
        prefix="all_scores",
    )

    # è una lista di tuple (prima tupla abbiamo gli integrated gradients del coso, seconda abbiamo i label grad) # per ogni esempio abbiamo se è confuso
    flag_list = list(itertools.chain([True for _ in range(len(wrong.ig_lists))], [False for _ in range(len(correct.ig_lists))]))
    grad_list = list(itertools.chain(wrong.ig_lists, correct.ig_lists))
    scatter_plot_score(grad_list, flag_list, arguments_folder, "All")
    scatter_plot_score(wrong.ig_lists, [True for _ in range(len(wrong.ig_lists))], arguments_folder, "Conf")
    scatter_plot_score(correct.ig_lists, [False for _ in range(len(correct.ig_lists))], arguments_folder, "Not conf")

    scatter_plot_score(correct.suitable_gradient_full_list, [True for _ in range(len(correct.suitable_gradient_full_list))], arguments_folder, "Suitable correct")
    scatter_plot_score(wrong.suitable_gradient_full_list, [False for _ in range(len(wrong.suitable_gradient_full_list))], arguments_folder, "Suitable wrong")

    tmp_suitable_list = list(itertools.chain([True for _ in range(len(correct.suitable_gradient_full_list))], [False for _ in range(len(wrong.suitable_gradient_full_list))]))
    scatter_plot_score(list(itertools.chain(correct.suitable_gradient_full_list, wrong.suitable_gradient_full_list)),tmp_suitable_list, arguments_folder, "Suitable")

    # only most suitable
    correct_guesses_conf = [True for _ in range(len(correct.suitable_ig_explaination_list))]
    wrongly_guesses_conf = [False for _ in range(len(wrong.suitable_ig_explaination_list))]
    input_gradient_scatter(
        correct_guesses=correct.suitable_ig_explaination_list,
        correct_guesses_conf=correct_guesses_conf,
        wrongly_guesses=wrong.suitable_ig_explaination_list,
        wrongly_guesses_conf=wrongly_guesses_conf,
        folder=arguments_folder,
        prefix="most_suitable_score",
    )

    correct_guesses_conf = [True for _ in range(len(correct.max_arguments_list))]
    wrongly_guesses_conf = [False for _ in range(len(wrong.max_arguments_list))]
    input_gradient_scatter(
        correct_guesses=correct.max_arguments_list,
        correct_guesses_conf=correct_guesses_conf,
        wrongly_guesses=wrong.max_arguments_list,
        wrongly_guesses_conf=wrongly_guesses_conf,
        folder=arguments_folder,
        prefix="max_score",
    )

    score_barplot(
        wrong.suitable_is_max,
        wrong.suitable_is_not_max,
        "Max",
        "Not Max",
        arguments_folder,
        "Suitable class is max or not max pred [confounded class]",
    )

    score_barplot(
        correct.suitable_is_max,
        correct.suitable_is_not_max,
        "Max",
        "Not Max",
        arguments_folder,
        "Suitable class is max or not max pred [not confounded class, corr class]",
    )

    score_barplot(
        wrong.influence_parent_counter,
        wrong.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Influence parent prediction vs not influence [conf]",
    )

    score_barplot(
        correct.influence_parent_counter,
        correct.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Influence parent prediction vs not influence [not conf]",
    )

    score_barplot(
        wrong.influence_parent_counter + correct.influence_parent_counter,
        wrong.not_influence_parent_counter + correct.not_influence_parent_counter,
        "Influence",
        "Not Influence",
        arguments_folder,
        "Influence parent prediction vs not influence",
    )

    score_barplot(
        correct.suitable_is_max + wrong.suitable_is_max,
        correct.suitable_is_not_max + wrong.suitable_is_not_max,
        "Max",
        "Not Max",
        arguments_folder,
        "Suitable class is max or not max pred",
    )

    max_arg_dictionary = correct.max_arguments_dict
    max_arg_dictionary.update(wrong.max_arguments_dict)
    label_arg_dictionary = correct.label_args_dict
    label_arg_dictionary.update(wrong.label_args_dict)
    plot_most_frequent_explainations(
        max_arg_dictionary, "Max chosen per class", arguments_folder
    )
    score_subclass_influence(
        label_arg_dictionary, "Label suitable per class", arguments_folder
    )


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
    arguments_folder: str
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
    Returns:
        ArgumentsStepArgs: arguments as class
    """


    bucket_list: List[ArgumentBucket] = list()
    table_correlation: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]] = dict()
    suitable_ig_explaination_list: List[float] = list()
    ig_lists: List[Tuple[List[float], List[float]]] = list()
    suitable_gradient_full_list: List[Tuple[List[float], List[float]]] = list()
    max_arguments_list: List[float] = list()
    max_arguments_dict: Dict[str, Dict[str, int]] = dict()
    label_args_dict: Dict[str, Dict[str, List[int]]] = dict()
    show_element_counter: int = 0
    influence_parent_counter: int = 0
    not_influence_parent_counter: int = 0
    suitable_is_max: int = 0
    suitable_is_not_max: int = 0

    # get samples
    selected_samples = get_samples_for_arguments(
        net=net,
        dataset=dataset,
        dataloaders=dataloaders,
        device=device,
        force_prediction=force_prediction,
        use_softmax=use_softmax,
        prediction_treshold=prediction_treshold,
        loader=dataloaders["test_loader_with_labels_name"],
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
            c_g, # sample
            net, # network
            labels_name, # label names
            dataloaders["test"].to_eval, # what to evaluate
            dataloaders["train_R"], # train R
            pred, # predicted integer labels
            groundtruth[0], # parent label
            groundtruth[1], # children label
        )

        # ig lists
        ig_lists.append(bucket.get_gradents_list_separated())
        # table correlation
        table_correlation = get_table_correlation_dictionary_from(bucket, correct_samples_only, table_correlation)
        # get the bucket in the bucket list
        bucket_list.append(bucket)
        # maximum score coming from the input gradienst
        max_score_ig: float = bucket.get_maximum_ig_score()

        # get the groundtruth label and the groundtruth gradient and score
        ground_truth_lab, ig_grad_score = bucket.get_ig_groundtruth()
        # add the score to the explaination list
        suitable_ig_explaination_list.append(ig_grad_score[1])
        # add the full gradients list
        suitable_gradient_full_list.append(bucket.get_gradents_list_separated_by_class(ground_truth_lab))

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
            ig_list, ig_titles = bucket.get_gradients_list_and_names()
            single_element_barplot(
                labels_name[ground_truth_lab], idx, ig_list, ig_titles, arguments_folder
            )
            number_element_to_show += 1

        if ig_grad_score[1] == max_score_ig:
            suitable_is_max += 1
        else:
            suitable_is_not_max += 1

        for int_label, ig_score_item in bucket.input_gradient_dict.items():
            str_groundtruth_label: str = labels_name[bucket.groundtruth_children]

            # add the max arguments
            if max_score_ig == ig_score_item[1]:
                max_arguments_list.append(ig_score_item[1])

            # populate the max arguments dictionary
            if not str_groundtruth_label in max_arguments_dict:
                max_arguments_dict[str_groundtruth_label] = {}
            tmp_dict = max_arguments_dict[str_groundtruth_label]
            if not labels_name[int_label] in tmp_dict:
                max_arguments_dict[str_groundtruth_label][labels_name[int_label]] = 0
            max_arguments_dict[str_groundtruth_label][labels_name[int_label]] += 1

        # coodo variables
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
        if bucket.label_gradient[i_c_l, i_p_l][1]:
            influence_parent_counter += 1
        else:
            not_influence_parent_counter += 1

    # return arguments
    return ArgumentsStepArgs(
        bucket_list = bucket_list,
        table_correlation = table_correlation,
        suitable_ig_explaination_list = suitable_ig_explaination_list,
        suitable_gradient_full_list = suitable_gradient_full_list,
        max_arguments_list = max_arguments_list,
        max_arguments_dict = max_arguments_dict,
        label_args_dict = label_args_dict,
        show_element_counter = show_element_counter,
        influence_parent_counter = influence_parent_counter,
        not_influence_parent_counter = not_influence_parent_counter,
        suitable_is_max = suitable_is_max,
        suitable_is_not_max = suitable_is_not_max,
        ig_lists = ig_lists
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
        plt.close()

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
        plt.xticks(rotation=0)
        plt.title("{} class {} total #{}".format(title, key, sum(max_arg_dictionary[key].values())))
        plt.tight_layout()
        fig.savefig("{}/{}_{}.png".format(folder, key, title))
        plt.close()


def scatter_plot_score(ig_list: List[Tuple[List[float], List[float]]], conf_list: List[bool], folder: str, prefix: str) -> None:
    """Method which plots two bars plot for each superclass, showing which class has the most frequent explainations
    Args:
        ig_list [List[Tuple[List[float], List[float]]]]: integrated gradient list
        conf_list [List[bool]]: confounded list
        folder [str]: folder
        prefix [str]: prefix
    """
    # è una lista di tuple (prima tupla abbiamo gli integrated gradients del coso, seconda abbiamo i label grad) # per ogni esempio abbiamo se è confuso
    rc = RidgeClassifier()
    score_list = list()
    label_list = list()
    for el, lab in zip(ig_list, conf_list):
        ig_grad, lab_grad = el
        for data in ig_grad:
            score_list.append([data])
            label_list.append(lab)
        for data in lab_grad:
            score_list.append([data])
            label_list.append(lab)
    score_list = np.array(score_list)
    label_list = np.array(label_list)

    rc.fit(score_list, label_list)
    score = rc.score(score_list, label_list)
    corr = scipy.stats.spearmanr(score_list, label_list)

    score_list = list()
    label_list = list()
    for el, lab in zip(ig_list, conf_list):
        ig_grad, lab_grad = el
        for data in ig_grad:
            score_list.append([0, data])
            label_list.append(lab)
        for data in lab_grad:
            score_list.append([1, int(data)])
            label_list.append(lab)
    score_list = np.array(score_list)
    label_list = np.array(label_list)
    rc.fit(score_list, label_list)

    # stacking the gradients with the x position => 0
    fig = plot_decision_surface(
        score_list,
        label_list,
        rc,
        "x",
        "gradient_magnitude",
        "Score: {}, spearman correlation {:.3f}, p-val {:.3f}\nsignificant with 95% confidence {} #conf {} #not-conf {}".format(
            score,
            corr[0],
            corr[1],
            corr[1] < 0.05,
            label_list.sum(),
            np.size(label_list) - label_list.sum(),
        ),
        True,
        0.001,
        max([el[1] for el in score_list]) + 0.1,
        False,
    )
    fig.savefig(
        "{}/{}_full_gradients_correlations.png".format(
            folder,
            prefix,
        )
    )


def plot_correlation_table(ig_list: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]) -> None:
    """Plots the correlation table given the integrated gradient list
    Args:
        ig_list: Dict[int, Dict[str, List[Tuple[bool, List[float]]]]]: integrated gradient list
    """
    for class_ in ig_list.keys():
        corr_text_dict = {}
        for lab in ig_list[class_].keys():
            corr_text_dict[lab] = scipy.stats.spearmanr(
                np.array([el[1] for el in ig_list[class_][lab]]),
                np.array([el[0] for el in ig_list[class_][lab]])
            )
            if np.isnan(corr_text_dict[lab].correlation):
                corr_text_dict[lab] = 0

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        df = pd.DataFrame([list(corr_text_dict.values())], columns=list(corr_text_dict.keys()))
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        fig.tight_layout()


def get_confounders(dataset) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Return the confounders given the dataset
    Args:
        dataset [str]
    Returns:
        Confounders list [Dict[str, Dict[str, List[Dict[str, Any]]]]]
    """
    # confounders
    confounders = cifar_confunders
    if dataset == "fashion":
        confounders = fashion_confunders
    elif dataset == "mnist":
        confounders = mnist_confunders
    elif dataset == "omniglot":
        confounders = omniglot_confunders
    return confounders


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

    # loop over the loader
    for _, inputs in tqdm.tqdm(enumerate(loader), desc="Return samples from dataloader"):
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
