import torch
import torchvision
import matplotlib.pyplot as plt
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from chmncc.dataset import load_cifar_dataloaders


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the dataset visualization
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("visualize", help="Dataset visualization subparser")
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="batch size of the datasets"
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


def visualize_train_datasets(
    train_loader: torch.utils.data.DataLoader,
    rows: int = 3,
    cols: int = 3,
) -> None:
    r"""
    Show the data from the dataloader

    **Note**: the number of images displayed depends on the size of the batch, since
    the bigger the batch size, the more the probability there are enogh samples of the
    desired label to be displayed.

    Default:

    - rows [int] = 3
    - cols [int] = 3

    Args:

    - train_loader [torch.utils.data.DataLoader]
    - rows [int]
    - cols [int]
    """

    # define iterators over both datasets
    train_iter = iter(train_loader)

    # get labels of source data
    data_source, _ = next(train_iter)

    if len(data_source) == 0:
        print("No data retrieved with given label")
        return

    # How many image it was able to retrieve
    print("Retreived {} images".format(len(data_source)))

    # source display
    display_grid = torchvision.utils.make_grid(
        data_source,
        nrow=rows,
        padding=2,
        pad_value=1,
        normalize=True,
        value_range=(data_source.min(), data_source.max()),
    )
    plt.imshow((display_grid.numpy().transpose(1, 2, 0)))
    plt.axis("off")
    plt.title(f"Train Dataset")
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
    dataloaders = load_cifar_dataloaders(
        img_size=32,
        img_depth=3,
        csv_path="./dataset/train.csv",
        test_csv_path="./dataset/train.csv",
        cifar_metadata="./dataset/pickle_files/meta",
        batch_size=10,
        test_batch_size=10,
        normalize=True,
    )

    # visualize
    print("Visualizing...")
    visualize_train_datasets(
        dataloaders["train_set"],  # train source loader
    )
