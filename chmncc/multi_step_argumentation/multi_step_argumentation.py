"""Model which provides the debugging facility to the model"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
import os
import torch.nn as nn
from chmncc.arguments.arguments_bucket import ArgumentBucket
from chmncc.dataset.load_dataset import LoadDataset
from chmncc.networks import ResNet18, LeNet5, LeNet7, AlexNet, MLP
from chmncc.utils.utils import (
    force_prediction_from_batch,
    load_best_weights,
    grouped_boxplot,
    plot_confusion_matrix_statistics,
    plot_global_multiLabel_confusion_matrix,
    get_confounders_and_hierarchy,
    prepare_dict_label_predictions_from_raw_predictions,
    plot_confounded_labels_predictions,
)
from chmncc.dataset import (
    load_dataloaders,
    get_named_label_predictions,
    LoadDebugDataset,
)
import wandb
from chmncc.test import test_step, test_step_with_prediction_statistics
from chmncc.optimizers import get_adam_optimizer, get_plateau_scheduler
from typing import Dict, Any, Tuple, Union, List
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torchsummary import summary
from itertools import tee
from chmncc.debug.debug import debug
import itertools
from enum import Enum


class ArgumentType(Enum):
    INPUT_GRADIENT = 1
    CLASS_GRADIENT = 2
    CLASS_ARGUMENT = 3
    UNKNOWN = 4


class AlgorithmOutcome(Enum):
    USER_DEFEAT = 1
    MACHINE_DEFEAT = 2
    GOING_ON = 3


class ArgumentsCounter:
    def __init__(self) -> None:
        self.label_counter: Dict[str, int] = {}

    def add(self, lab: str) -> None:
        if lab not in self.label_counter:
            self.label_counter[lab] = 0
        self.label_counter[lab] += 1

    def _increase(self, lab: str, val: int) -> None:
        if lab not in self.label_counter:
            self.label_counter[lab] = 0
        self.label_counter[lab] += val

    def update(self, argument_counter) -> None:
        tmp_counter: Dict[str, int] = argument_counter.label_counter
        for lab in tmp_counter:
            self._increase(lab, tmp_counter[lab])


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the debug of the network
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser(
        "multi-step-argumentation", help="Debug network subparser"
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["resnet", "lenet", "lenet7", "alexnet", "mlp"],
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
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
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
        "--prediction-treshold",
        type=float,
        default=0.5,
        help="considers the class to be predicted in a multilabel classification setting",
    )
    parser.add_argument("--patience", type=int, default=5, help="scheduler patience")
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
    parser.add_argument("--norm-exponent", type=int, default=2, help="norm exponent")
    parser.add_argument("--tau", type=float, default=0.0, help="tau")
    parser.add_argument(
        "--max-rounds-per-iterations",
        type=int,
        default=10,
        help="maximum rounds per iterations",
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(
        func=main,
        constrained_layer=True,
        force_prediction=False,
        fixed_confounder=False,
        use_softmax=False,
        simplified_dataset=False,
        imbalance_dataset=False,
    )


def multi_step_argumentation(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    iterations: int,
    device: str,
    test_batch_size: int,
    num_workers: int,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    labels_names: List[str],
    max_rounds_per_iterations: int,
    dataset_type: str,
    tau: float,
    norm_exponent: int,
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
        **kwargs [Any]: kwargs
    """
    print(
        "Have to run for the multi-step argumentation procedure for {} iterations...".format(
            iterations
        )
    )

    ## ==================================================================
    # Load debug datasets: for training the data -> it has labels and confounders position information
    test_set = LoadDebugDataset(
        dataloaders["test_dataset_with_labels_and_confunders_position"],
    )

    print("Debug dataset statistics:")
    test_set.train_set.print_stats()

    # Dataloaders for the previous values
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    # network in evaluation mode
    net.eval()

    # new data list
    new_data_list: List[Tuple[np.ndarray, str, str]] = list()
    argument_counter: ArgumentsCounter = ArgumentsCounter()

    # steps
    for step, data in enumerate(iter(test_loader)):
        (
            inputs,
            hierarchical_labels,
            confounder_masks,
            confounded_list,
            superclass_list,
            subclass_list,
        ) = data

        # load data into device
        inputs = inputs.to(device)
        hierarchical_labels = hierarchical_labels.to(device)
        hierarchical_labels = torch.unsqueeze(hierarchical_labels, 0)
        hierarchical_labels = hierarchical_labels > 0.5

        # output of the network
        logits = net(inputs.float())

        # force prediction
        if force_prediction:
            predicted_1_0 = force_prediction_from_batch(
                logits.data.cpu(),
                prediction_treshold,
                use_softmax,
                test_set.train_set.n_superclasses,
            )
        else:
            predicted_1_0 = logits.data.cpu() > prediction_treshold  # 0.5

        processed_samples_list, process_samples_counter = process_samples(
            inputs=inputs,
            hierarchical_labels=hierarchical_labels.squeeze(),
            confounder_masks=confounder_masks.squeeze(),
            confounded_list=confounded_list.squeeze(),
            superclass_list=superclass_list,
            subclass_list=subclass_list,
            predicted_1_0=predicted_1_0.squeeze(),
            test_set=test_set.train_set,
            test_set_R=dataloaders["train_R"],
            net=net,
            labels_names=labels_names,
            max_rounds_per_iterations=max_rounds_per_iterations,
            dataset=dataset_type,
            tau=tau,
            norm_exponent=norm_exponent,
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
        )

        # extend list
        new_data_list.extend(processed_samples_list)
        argument_counter.update(process_samples_counter)

    # TODO vedere come bilanciare. Magari fare la conta?
    debug_to_use = LoadDebugDataset(
        dataloaders["test_dataset_with_labels_and_confunders_position"],
    )
    debug_to_use.train_set.data_list = new_data_list


def process_samples(
    inputs: torch.Tensor,
    hierarchical_labels: torch.Tensor,
    confounder_masks: torch.Tensor,
    confounded_list: List[bool],
    superclass_list: List[str],
    subclass_list: List[str],
    predicted_1_0: torch.Tensor,
    test_set: LoadDataset,
    test_set_R: torch.Tensor,
    net: nn.Module,
    labels_names: List[str],
    max_rounds_per_iterations: int,
    dataset: str,
    tau: float,
    norm_exponent: int = 2,
    prediction_treshold: float = 0.5,
    force_prediction: bool = False,
    use_softmax: bool = False,
) -> Tuple[List[Tuple[np.ndarray, str, str]], ArgumentsCounter]:

    new_data_list: List[Tuple[np.ndarray, str, str]] = list()
    argument_counter: ArgumentsCounter = ArgumentsCounter()

    for i, (
        single_el,
        label,
        mask,
        confounded,
        superclass,
        subclass,
        pred_item,
    ) in enumerate(
        zip(
            inputs,
            hierarchical_labels,
            confounder_masks,
            confounded_list,
            superclass_list,
            subclass_list,
            predicted_1_0,
        )
    ):

        # detect wrongly predicted samples
        if not torch.equal(
            pred_item[test_set.to_eval],
            label[test_set.to_eval],
        ):
            # get the named prediction
            named_prediction = get_named_label_predictions(
                pred_item.to(torch.float),
                test_set.get_nodes(),
            )

            print("Named prediction", named_prediction)

            # extract parent and children
            confunders, children, parents = get_confounders_and_hierarchy(dataset)

            # children and parent predictions
            children_prediction: List[str] = []
            parent_prediction: List[str] = []

            children = list(itertools.chain.from_iterable(list(children)))
            parents = list(parents)

            for pred in named_prediction:
                if pred in children:
                    children_prediction.append(pred)
                else:
                    parent_prediction.append(pred)

            # np image
            npimg = single_el.detach().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
            plt.title(
                "Groundtruth: {} > {}\nPredicted {} because machine saw {}".format(
                    superclass, subclass, parent_prediction, children_prediction
                )
            )
            plt.show()
            plt.close()

            predicted_list = (
                (pred_item[test_set.to_eval] == True).nonzero().flatten().tolist()
            )

            print("Predicted list", predicted_list)

            bucket = ArgumentBucket(
                single_el,  # sample
                net,  # network
                labels_names,  # label names
                test_set.to_eval,  # what to evaluate
                test_set_R,  # train R
                predicted_list,  # predicted integer labels
                superclass,  # parent label
                subclass,  # children label
                norm_exponent,  # norm exponent
                prediction_treshold,  # prediction threshold
                force_prediction,  # force prediction
                use_softmax,  # use softmax
            )

            outcome, tmp_argument_counter = arguments_loop(
                max_rounds_per_iterations,
                named_prediction,
                bucket,
                parents,
                tau,
                labels_names,
            )

            # model needs to be updated
            if outcome == AlgorithmOutcome.USER_DEFEAT:
                print("The model does not need to be updated")
            elif outcome == AlgorithmOutcome.MACHINE_DEFEAT:
                print("The model needs to be updated, maybe not every single step")
                new_data_list.append((single_el.detach().numpy(), superclass, subclass))
                argument_counter.update(tmp_argument_counter)

    # return new datalist
    return new_data_list, argument_counter


def ask_defeat() -> bool:
    user_input = ""

    user_input = input("Do you want to give up? yes/no: ")
    if user_input.lower() == "yes":
        return True
    elif user_input.lower() == "no":
        print("User typed no")
        return False
    print("Assumed the user has typed no")
    return False


def remove_maximum_argument(
    arguments_list: Tuple[
        Dict[int, Tuple[float, torch.Tensor]],
        Dict[Tuple[int, int], Tuple[float, torch.Tensor]],
    ]
) -> Tuple[
    Tuple[Union[int, Tuple[int, int]], float, torch.Tensor],
    ArgumentType,
    Tuple[
        Dict[int, Tuple[float, torch.Tensor]],
        Dict[Tuple[int, int], Tuple[float, torch.Tensor]],
    ],
]:
    max_key: Union[Tuple[int, int], int] = -1
    max_score: float = -float("inf")
    max_grad: torch.Tensor = torch.zeros((1, 28, 28))
    arg_type: ArgumentType = ArgumentType.UNKNOWN

    for key, val in arguments_list[0].items():
        (score, grad) = val
        if score > max_score:
            max_score = score
            max_key = key
            max_grad = grad
            arg_type = ArgumentType.INPUT_GRADIENT

    for key, val in arguments_list[1].items():
        (score, grad) = val
        if score > max_score:
            max_score = score
            max_grad = grad
            max_key = key
            arg_type = ArgumentType.CLASS_GRADIENT

    index_remove = 0
    if type(max_key) is tuple:
        index_remove = 1

    del arguments_list[index_remove][max_key]

    return (max_key, max_score, max_grad), arg_type, arguments_list


def display_input_gradient_argument(
    image: np.ndarray, gradient: torch.Tensor, title: str
):
    print(np.max(gradient.detach().numpy().flatten()))
    print(gradient.detach().numpy().flatten().shape)
    norm = matplotlib.colors.Normalize(
        vmin=0, vmax=np.max(gradient.detach().numpy().flatten())
    )
    # show the picture
    plt.colorbar(
        matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm),
        label="Gradient magnitude",
    )
    plt.imshow(np.transpose(image, (1, 2, 0)), cmap="gray")
    plt.imshow(np.transpose(gradient, (1, 2, 0)), cmap="viridis", alpha=0.5)
    plt.title("{} gradient overlay".format(title))
    plt.show()
    plt.close()


def arguments_loop(
    max_rounds: int,
    named_prediction: List[str],
    arguments_bucket: ArgumentBucket,
    parents: List[str],
    tau: float,
    labels_names: List[str],
) -> Tuple[AlgorithmOutcome, ArgumentsCounter]:
    argument_counter: ArgumentsCounter = ArgumentsCounter()
    outcome: AlgorithmOutcome = AlgorithmOutcome.GOING_ON
    arguments_list: List[
        Tuple[ArgumentType, Union[List[int], Tuple[int, int], int], float, bool]
    ] = list()
    machine_arguments_list: Tuple[
        Dict[int, Tuple[float, torch.Tensor]],
        Dict[Tuple[int, int], Tuple[float, torch.Tensor]],
    ] = arguments_bucket.get_arguments_lists_separated_by_prediction(parents)
    user_turn: bool = True

    print("#> [Sample idx 0]: arg_0^m = {}".format(named_prediction))
    arguments_list.append(
        (ArgumentType.CLASS_ARGUMENT, arguments_bucket.guess_list, -1, False)
    )

    # for a certain number of turns
    for t in range(1, max_rounds):

        # outcome finish
        if outcome != AlgorithmOutcome.GOING_ON:
            break

        # user turn
        if user_turn:
            # user turn
            if not ask_defeat():
                # image the user uses a class argument
                arg_class_int = int(
                    input(
                        "#> [Sample idx {}]: arg_{}^u [type: class] yi = ".format(t, t)
                    )
                )
                # add the user argument
                arguments_list.append(
                    (ArgumentType.CLASS_ARGUMENT, arg_class_int, -1, True)
                )
            else:
                print("User admits defeat")
                outcome = AlgorithmOutcome.USER_DEFEAT
        else:
            # machine turn
            (
                (arg_key, arg_score, arg_grad),
                arg_type,
                machine_arguments_list,
            ) = remove_maximum_argument(machine_arguments_list)

            # machine declares defeat
            if arg_score < tau:
                print("Machine admits defeat")
                outcome = AlgorithmOutcome.MACHINE_DEFEAT
                break

            # string to display
            str_to_display = "#> [Sample idx {}]: arg_{}^d [type: {}] = ".format(
                t, t, ArgumentType(arg_type).name
            )

            # select among argument types
            if arg_type == ArgumentType.INPUT_GRADIENT:
                str_to_display = "{} input gradient wrt to {}, score: {}".format(
                    str_to_display, labels_names[arg_key], arg_score
                )
                display_input_gradient_argument(
                    arguments_bucket.sample.detach().numpy(),
                    arg_grad,
                    "input gradient wrt to {}, score: {:3f}".format(
                        labels_names[arg_key], arg_score
                    ),
                )
            elif arg_type == ArgumentType.CLASS_GRADIENT:
                str_to_display = "{} class gradient: predicted {} because machine saw {}, score: {}".format(
                    str_to_display,
                    labels_names[arg_key[1]],
                    labels_names[arg_key[0]],
                    arg_score,
                )
                argument_counter.add(labels_names[arg_key[1]])

            # add the user argument
            arguments_list.append((arg_type, arg_key, arg_score, False))

            # string to display
            print(str_to_display)

        # change turn
        user_turn = not user_turn

    print("At this point, update the model...")
    return outcome, argument_counter


def main(args: Namespace) -> None:
    """Checks the command line arguments and then runs the multi-step argumentation debug

    Args:
        args (Namespace): command line arguments
    """
    print("\n### Network debug ###")
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
        imbalance_dataset=args.imbalance_dataset,
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
    # zero grad
    net.zero_grad()
    # summary
    summary(net, (img_depth, img_size, img_size))

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

    # optimizer
    optimizer = get_adam_optimizer(
        net, args.learning_rate, weight_decay=args.weight_decay
    )
    # scheduler
    scheduler = get_plateau_scheduler(optimizer=optimizer, patience=args.patience)

    # Test on best weights (of the confounded model)
    load_best_weights(net, args.weights_path_folder, args.device)

    # dataloaders
    test_loader = dataloaders["test_loader"]

    # define the cost function (binary cross entropy for the current models)
    cost_function = torch.nn.BCELoss()

    # test set
    #  test_loss, test_accuracy, test_score_raw, test_score_const, _, _ = test_step(
    #      net=net,
    #      test_loader=iter(test_loader),
    #      cost_function=cost_function,
    #      title="Test",
    #      test=dataloaders["test"],
    #      device=args.device,
    #      prediction_treshold=args.prediction_treshold,
    #      force_prediction=args.force_prediction,
    #      superclasses_number=dataloaders["train_set"].n_superclasses,
    #  )
    #
    labels_name = dataloaders["test_set"].nodes_names_without_root
    #  test_loader_with_label_names = dataloaders["test_loader_with_labels_name"]
    #
    #  print("Network resumed, performances:")
    #
    #  print(
    #      "\n\t [TEST SET]: Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve raw {:.3f}, Test Area under Precision-Recall Curve const {:.3f}".format(
    #          test_loss, test_accuracy, test_score_raw, test_score_const
    #      )
    #  )

    print("-----------------------------------------------------")

    print("#> Multi-step argumentation...")

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.watch(net)

    # launch the debug a given number of iterations
    multi_step_argumentation(
        net=net,
        dataloaders=dataloaders,
        iterations=1,
        device=args.device,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        prediction_treshold=args.prediction_treshold,
        force_prediction=args.force_prediction,
        use_softmax=args.use_softmax,
        labels_names=labels_name,
        max_rounds_per_iterations=args.max_rounds_per_iterations,
        dataset_type=args.dataset,
        tau=args.tau,
        norm_exponent=args.norm_exponent,
    )

    # lancia il debug loop
    debug(
        net,
        dataloaders,
        debug_folder,
        iterations,
        cost_function,
        device,
        set_wandb,
        integrated_gradients,
        optimizer,
        scheduler,
        debug_test_loader,
        batch_size,
        test_batch_size,
        reviseLoss,
        model_folder,
        network,
        gradient_analysis,
        num_workers,
        prediction_treshold,
        force_prediction,
        use_softmax,
        dataset,
        balance_subclasses,
        balance_weights,
        correct_by_duplicating_samples,
        to_correct_dataloader,
    )

    exit(0)

    print("After correction...")

    # re-test set
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
    )

    print(
        "\n\t [TEST SET]: Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve raw {:.3f}, Test Area under Precision-Recall Curve const {:.3f}".format(
            test_loss, test_accuracy, test_score_raw, test_score_const
        )
    )

    # collect stats
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
        _,
    ) = test_step_with_prediction_statistics(
        net=net,
        test_loader=iter(test_loader_with_label_names),
        cost_function=cost_function,
        title="Collect Statistics [TEST]",
        test=dataloaders["test"],
        device=args.device,
        labels_name=labels_name,
        prediction_treshold=args.prediction_treshold,
        force_prediction=args.force_prediction,
        superclasses_number=dataloaders["train_set"].n_superclasses,
    )

    ## confusion matrix after debug
    plot_global_multiLabel_confusion_matrix(
        y_test=y_test,
        y_est=y_pred,
        label_names=labels_name,
        size=(30, 20),
        fig_name="{}/test_after_confusion_matrix_normalized".format(args.debug_folder),
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
        fig_name="{}/test_after_confusion_matrix_statistics.png".format(
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
        test_loader=iter(
            dataloaders["test_loader_only_label_confounders_with_labels_names"]
        ),
        cost_function=cost_function,
        title="Computing statistics in label confoundings",
        test=dataloaders["train"],
        device=args.device,
        labels_name=labels_name,
        prediction_treshold=args.prediction_treshold,
        force_prediction=args.force_prediction,
        use_softmax=args.use_softmax,
        superclasses_number=dataloaders["train_set"].n_superclasses,
    )

    (
        labels_predictions_dict,
        counter_dict,
    ) = prepare_dict_label_predictions_from_raw_predictions(
        y_pred, y_test, labels_name, args.dataset, True
    )
    plot_confounded_labels_predictions(
        labels_predictions_dict,
        counter_dict,
        args.debug_folder,
        "imbalancing_predictions",
        args.dataset,
    )

    # save the model state of the debugged network
    torch.save(
        net.state_dict(),
        os.path.join(
            args.model_folder,
            "after_training_multi_step_correction_{}_{}.pth".format(
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
