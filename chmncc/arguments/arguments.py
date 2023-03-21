"""Analyze arguments"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
import os
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from chmncc.networks import ResNet18, LeNet5, LeNet7, AlexNet, MLP
from chmncc.config import (
    cifar_hierarchy,
    mnist_hierarchy,
    fashion_hierarchy,
    omniglot_hierarchy,
)
from chmncc.config import (
    cifar_confunders,
    cifar_hierarchy,
    mnist_hierarchy,
    mnist_confunders,
    fashion_hierarchy,
    fashion_confunders,
    omniglot_hierarchy,
    omniglot_confunders,
)
from chmncc.utils.utils import (
    force_prediction_from_batch,
    load_last_weights,
    load_best_weights,
    grouped_boxplot,
    plot_confusion_matrix_statistics,
    plot_global_multiLabel_confusion_matrix,
    get_lr,
    dotdict,
)
from chmncc.dataset import (
    load_dataloaders,
    get_named_label_predictions,
    LoadDebugDataset,
)
from chmncc.explanations import compute_integrated_gradient, output_gradients
from chmncc.optimizers import get_adam_optimizer, get_plateau_scheduler
from typing import Dict, Any, Tuple, Union, List
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torchsummary import summary
from sklearn.linear_model import RidgeClassifier
from chmncc.arguments.arguments_bucket import ArgumentBucket


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


def arguments(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    arguments_folder: str,
    iterations: int,
    device: str,
    model_folder: str,
    prediction_treshold: float,
    force_prediction: bool,
    use_softmax: bool,
    dataset: str,
    **kwargs: Any
) -> None:

    # load the training dataloader
    training_loader_with_labels_names = dataloaders["training_loader_with_labels_names"]

    # load the human readable labels dataloader
    test_loader_with_label_names = dataloaders["test_loader_with_labels_name"]

    # correctly guessed elements
    correctly_guessed: List[torch.Tensor] = list()
    wrongly_correctly_guessed: List[torch.Tensor] = list()

    # labels name
    labels_name = dataloaders["test_set"].nodes_names_without_root

    print("Have to run for {} arguments iterations...".format(iterations))

    # running for the requested iterations
    for it in range(iterations):

        print("Start iteration number {}".format(it))
        print("-----------------------------------------------------")

        correctly_guessed = get_samples_for_arguments(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            loader=training_loader_with_labels_names,
            correct=True,
            num_element_to_analyze=2,
            test=dataloaders["test"],
        )

        for idx, c_g in enumerate(correctly_guessed):
            bucket = ArgumentBucket(
                c_g,
                net,
                labels_name,
                dataloaders["test"].to_eval,
                dataloaders["train_R"],
            )
            plt.title("Correct: Number {}".format(idx))
            plt.imshow(c_g.detach().numpy().transpose(1, 2, 0), cmap="gray")
            plt.show()

            print("Bucket", bucket)

        wrongly_guessed = get_samples_for_arguments(
            net=net,
            dataset=dataset,
            dataloaders=dataloaders,
            device=device,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            prediction_treshold=prediction_treshold,
            loader=test_loader_with_label_names,
            correct=False,
            num_element_to_analyze=2,
            test=dataloaders["test"],
        )

        for idx, c_g in enumerate(wrongly_guessed):
            bucket = ArgumentBucket(
                c_g,
                net,
                labels_name,
                dataloaders["test"].to_eval,
                dataloaders["train_R"],
            )
            plt.title("Wrong: Number {}".format(idx))
            plt.imshow(c_g.detach().numpy().transpose(1, 2, 0), cmap="gray")
            plt.show()

            print("Bucket", bucket)


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
) -> List[torch.Tensor]:
    # datalist
    data_list: List[torch.Tensor] = list()
    # completed
    completed = False
    # loop over the loader
    for batch_idx, inputs in tqdm.tqdm(enumerate(loader), desc="Save sample in loader"):
        # according to the Giunchiglia dataset
        (inputs, superclass, subclass, targets) = inputs

        for el_idx in range(inputs.shape[0]):
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
            # get the named prediction
            named_prediction = get_named_label_predictions(
                predicted_1_0, dataloaders["test_set"].get_nodes()
            )
            # extract parent and children
            parents = cifar_hierarchy.keys()
            children = cifar_hierarchy.values()
            # switch on dataset
            if dataset == "mnist":
                children = mnist_hierarchy.values()
                parents = mnist_hierarchy.keys()
            elif dataset == "fashion":
                children = fashion_hierarchy.values()
                parents = fashion_hierarchy.keys()
            elif dataset == "omniglot":
                children = omniglot_hierarchy.values()
                parents = omniglot_hierarchy.keys()
            # get the name prediction on the hierarchy
            children = [
                element for element_list in children for element in element_list
            ]
            parent_predictions = list(filter(lambda x: x in parents, named_prediction))
            children_predictions = list(
                filter(lambda x: x in children, named_prediction)
            )
            # select whether it is confunded
            #  print("Ground-truth:", superclass[el_idx], subclass[el_idx])
            #  print("Predicted", parent_predictions, children_predictions)
            #  print("Predicted 0/1:", predicted_1_0)
            #  print("Single target", single_target)
            print("Ground-truth:", superclass[el_idx], subclass[el_idx])
            print("Predicted:", parent_predictions, children_predictions)
            #  print("Predicted 0/1:", predicted_bool)
            #  print("Single target", single_target_bool)
            import time

            time.sleep(1)
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
                print("added!")
                data_list.append(inputs[el_idx])

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
