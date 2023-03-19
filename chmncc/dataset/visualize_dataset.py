import torch
import torchvision
import matplotlib.pyplot as plt
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from chmncc.dataset import load_dataloaders, get_named_label_predictions
from chmncc.config import (
    cifar_confunders,
    mnist_confunders,
    fashion_confunders,
    omniglot_confunders,
)
from typing import List


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the dataset visualization
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("visualize", help="Dataset visualization subparser")
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=12, help="batch size of the datasets"
    )
    parser.add_argument(
        "--confunder",
        "-c",
        type=bool,
        default=True,
        help="whether to show confunders according to the config",
    )
    parser.add_argument(
        "--train",
        "-t",
        type=bool,
        default=True,
        help="whether to show the dataset retrieved in training mode",
    )
    parser.add_argument(
        "--only-confounders",
        "-oc",
        type=bool,
        default=False,
        help="whether to show images with confounder only",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="cifar",
        choices=["cifar", "mnist", "fashion", "omniglot"],
        help="whether to use mnist dataset",
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


def visualize_train_datasets(
    train_loader: torch.utils.data.DataLoader,
    nodes: List[str],
    phase: str,
    num_images: int,
    dataset: str,
    rows: int = 4,
    only_confounders: bool = False,
) -> None:
    r"""
    Show the data from the dataloader

    **Note**: the number of images displayed depends on the size of the batch, since
    the bigger the batch size, the more the probability there are enogh samples of the
    desired label to be displayed.

    Default:
        rows [int] = 4
        only_confunders [bool] = False

    Args:
        train_loader [torch.utils.data.DataLoader]
        nodes [List[str]]: list of nodes names
        phase [str]: which phase we are in (test or train)
        num_images [int]: number of images to retrieved
        rows [int]
        only_confunders [bool]: whether to show only confunded images
    """

    train_iter = iter(train_loader)

    confounders = cifar_confunders
    if dataset == "mnist":
        confounders = mnist_confunders
    elif dataset == "fashion":
        confounders = fashion_confunders
    elif dataset == "omniglot":
        confounders = omniglot_confunders

    if only_confounders:
        data_source, labels = [], []
        # fill the data
        try:
            while len(data_source) < num_images:
                tmp_data_source, tmp_labels = next(train_iter)
                label_names = [
                    get_named_label_predictions(tmp_labels[i], nodes)
                    for i in range(tmp_labels.shape[0])
                ]
                # filter data
                for i in range(len(label_names)):
                    superclass, subclass = label_names[i]
                    # skip the data not confunded
                    if not superclass in confounders:
                        continue
                    # set the data
                    for j in range(len(confounders[superclass][phase])):
                        # skip invalid subclasses
                        if (
                            not subclass
                            in confounders[superclass][phase][j]["subclass"]
                        ):
                            continue
                        # add the correct data
                        data_source.append(tmp_data_source[i])
                        labels.append(tmp_labels[i])
                        # exit when the dimension is ok
                        if len(data_source) == num_images:
                            break
        except StopIteration:
            pass
        # to tensor
        data_source = torch.stack(data_source)
        labels = torch.stack(labels)
    else:
        data_source, labels = next(train_iter)

    if len(data_source) == 0:
        print("No data retrieved with given label")
        return

    # How many image it was able to retrieve
    print("Retreived {} images".format(len(data_source)))

    print("Labels:")
    for i in range(labels.shape[0]):
        named_labels = get_named_label_predictions(labels[i], nodes)
        print("Image {} has labels {}: ".format(i, named_labels))

    # source display
    display_grid = torchvision.utils.make_grid(
        data_source,
        nrow=rows,
        padding=2,
        pad_value=1,
        normalize=False,
        value_range=(data_source.min(), data_source.max()),
    )

    plt.imshow((display_grid.numpy().transpose(1, 2, 0)))
    plt.axis("off")
    plt.title("Dataset")
    plt.tight_layout()
    plt.show()


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the dataset visualization
    Args:
      args (Namespace): command line arguments
    """
    print("\n### Dataset visualization ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # Load dataloaders
    print("Load dataloaders...")
    dataloaders = load_dataloaders(
        dataset_type=args.dataset,
        img_size=32,
        img_depth=3,
        csv_path="./dataset/train.csv",
        test_csv_path="./dataset/test_reduced.csv",
        val_csv_path="./dataset/val.csv",
        cifar_metadata="./dataset/pickle_files/meta",
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        normalize=False,
        device="cpu",
        confunder=args.confunder,
    )

    dataloader = dataloaders["train_loader"]  # train source loader
    phase = "train"
    nodes = dataloaders["train_set"].get_nodes()
    if not args.train:
        phase = "test"
        dataloader = dataloaders["test_loader"]  # test source loader
        nodes = dataloaders["test_set"].get_nodes()

    # visualize
    print("Visualizing...")
    visualize_train_datasets(
        dataloader,
        nodes,
        phase,
        args.batch_size,
        only_confounders=args.only_confounders,
        dataset=args.dataset,
    )
