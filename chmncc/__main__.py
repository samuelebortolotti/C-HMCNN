"""Main module of the `c-hmcnn` project

This code has been developed by Eleonora Giunchiglia and Thomas Lukasiewicz; later modified by Samuele Bortolotti
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import torch
import tqdm
import cv2 as cv
import torch.nn as nn
from argparse import Namespace
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from typing import Any
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import wandb
import signal

# data folder
os.environ["DATA_FOLDER"] = "./"
os.environ["MODELS"] = "./models"
os.environ["IMAGE_FOLDER"] = "./plots"

from chmncc.utils.utils import (
    load_best_weights,
    log_values,
    load_best_weights,
    resume_training,
    get_lr,
)
from chmncc.networks.ConstrainedFFNN import initializeConstrainedFFNNModel
from chmncc.networks import LeNet5, ResNet18
from chmncc.train import training_step
from chmncc.optimizers import get_adam_optimizer
from chmncc.test import test_step
from chmncc.dataset import load_old_dataloaders, load_cifar_dataloaders
import chmncc.dataset.preproces_cifar as data
import chmncc.dataset.visualize_dataset as visualize_data
from chmncc.config.old_config import lrs, epochss, hidden_dims
from chmncc.explanations import compute_integrated_gradient, output_gradients


class TerminationError(Exception):
    """
    Error raised when a termination signal is received
    """

    def __init__(self):
        super().__init__("External signal received: forcing termination")


def __handle_signal(signum: int, frame):
    raise TerminationError()


def register_termination_handlers():
    """
    Makes this process catch SIGINT and SIGTERM.
    When the process receives such a signal after this call, a TerminationError is raised.
    """

    signal.signal(signal.SIGINT, __handle_signal)
    signal.signal(signal.SIGTERM, __handle_signal)


def get_args() -> Namespace:
    """Parse command line arguments.

    Returns:
      Namespace: command line arguments
    """
    # main parser
    parser = argparse.ArgumentParser(
        prog="Coherent Hierarchical Multi-Label Classification Networks",
        description="""Hierarchical Multi-Label Classification with explainations""",
    )

    # subparsers
    subparsers = parser.add_subparsers(help="sub-commands help")
    configure_subparsers(subparsers)
    # configure the dataset subparser
    data.configure_subparsers(subparsers)
    visualize_data.configure_subparsers(subparsers)

    # parse the command line arguments
    parsed_args = parser.parse_args()

    # if function not passed, then print the usage and exit the program
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)

    return parsed_args


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the coherent hierarchical multi-label classification
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser(
        "experiment",
        help="Coherent Hierarchical Multi-Label Classification experiment helper",
    )

    # Required  parameter
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help='dataset name, must end with: "_GO", "_FUN", or "_others"',
    )

    # lascio
    parser.add_argument("exp_name", type=str, help="name of the experiment")
    parser.add_argument("epochs", type=int, help="number of training epochs")
    # Other parameters
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--device", type=str, default="gpu:0", help="GPU (default:0)")

    parser.add_argument(
        "--giunchiglia",
        type=bool,
        default=False,
        help="apply Giunchiglia et al. approach",
    )

    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decy")
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=20,
        help="how frequent to save the model",
    )
    parser.add_argument(
        "--dry", type=bool, default=False, help="do not save checkpoints"
    )
    parser.add_argument(
        "--project", "-w", type=str, default="chmcnn-project", help="wandb project"
    )
    parser.add_argument(
        "--entity", "-e", type=str, default="samu32", help="wandb entity"
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["lenet", "resnet"],
        default="lenet",
        help="CNN architecture",
    )
    parser.add_argument(
        "--pretrained", type=bool, default=False, help="load the pretrained model"
    )
    parser.add_argument("--wandb", "-wdb", type=bool, default=False, help="wandb")
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=experiment)


def c_hmcnn(
    exp_name: str,
    resume: bool = False,
    device: str = "cuda",
    batch_size: int = 128,
    test_batch_size: int = 128,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-5,
    epochs: int = 30,
    save_every_epochs: int = 10,
    dry: bool = False,
    set_wandb: bool = False,
    dataset: str = "",
    network: str = "lenet",
    pretrained: bool = False,
    **kwargs: Any,
) -> None:
    r"""
    Function which performs both training and test step

    - resume [bool] = False: by default do not resume last training
    - device [str] = "cuda": move tensors on GPU, could be also "cpu"
    - batch_size [int] = 128
    - test_batch_size [int] = 128
    - learning_rate [float] = 0.01
    - epochs [int] = 30
    - weight_decay [float] = 1e-5
    - save_every_epochs [int] = 10: save a checkpoint every 10 epoch
    - dry [bool] = False: by default save weights
    - set_wandb [bool] = False,
    - dataset [str] = "", taken for retrocompatibility with Giunchiglia et al approach
    - network [str] = "lenet"
    - pretrained [bool] = False

    Args:

    - exp_name [str]: name of the experiment, basically where to save the logs of the SummaryWriter
    - resume [bool] = False: whether to resume a checkpoint
    - device [str] = "cuda": where to load the tensors
    - batch_size [int] = 128: default batch size
    - test_batch_size [int] = 128: default batch size for the test set
    - learning_rate [float] = 0.01: initial learning rate
    - weight_decay [float] = 1e-5: weigt decay
    - epochs [int] = 30: number of epochs
    - save_every_epochs: int = 10: save a checkpoint every `save_every_epochs` epoch
    - dry [bool] = False: whether to do not save weights
    - set_wandb [bool] = False: whether to log values on wandb
    - dataset [str] = str: dataset name: the dataset is specified -> old approach
    - network [str] = "lenet": which arachitecture to employ
    - pretrained [bool] = False, whether the network is pretrained [Note: lenet is not pretrained]
    - \*\*kwargs [Any]: additional key-value arguments
    """

    # create the models directory
    model_folder = os.environ["MODELS"]
    # create folders for the project
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(os.environ["IMAGE_FOLDER"], exist_ok=True)

    log_directory = "runs/exp_{}".format(exp_name)

    # old method
    old_method = kwargs.pop("giunchiglia")

    # create a logger for the experiment
    writer = SummaryWriter(log_dir=log_directory)

    # create folder for the experiment
    os.makedirs(exp_name, exist_ok=True)

    # Set up the metrics
    metrics = {
        "loss": {"train": 0.0, "val": 0.0, "test": 0.0},
        "acc": {"train": 0.0, "val": 0.0, "test": 0.0},
        "score": {"val": 0.0, "test": 0.0},
    }

    # get dataloaders
    if old_method:
        # Load the datasets
        dataloaders = load_old_dataloaders(dataset, batch_size, device=device)
    else:
        dataloaders = load_cifar_dataloaders(
            img_size=32,  # the size is squared
            img_depth=3,  # number of channels
            device=device,
            csv_path="./dataset/train.csv",
            test_csv_path="./dataset/test_reduced.csv",
            val_csv_path="./dataset/val.csv",
            cifar_metadata="./dataset/pickle_files/meta",
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            normalize=True,  # nrmalize the dataset
        )

    # network initialization
    if old_method:  # Giunchiglia et al method
        data = dataset.split("_")[0]  # first part is the data
        ontology = dataset.split("_")[1]  # second part is the ontology

        # hyperparams
        hyperparams = {
            "batch_size": batch_size,
            "num_layers": 3,
            "dropout": 0.7,
            "non_lin": "relu",
            "hidden_dim": hidden_dims[ontology][data],
            "lr": lrs[ontology][data],
            "weight_decay": weight_decay,
        }

        print("current hyperparams:", hyperparams)

        # MPL
        net = initializeConstrainedFFNNModel(
            dataset, data, ontology, dataloaders["train_R"], hyperparams
        )
    else:
        # CNN
        if network == "lenet":
            net = LeNet5(
                dataloaders["train_R"], 121
            )  # 20 superclasses, 100 subclasses + the root
        else:
            net = ResNet18(
                dataloaders["train_R"], 121, pretrained
            )  # 20 superclasses, 100 subclasses + the root

    # move the network
    net = net.to(device)

    print("#> Model")
    summary(net, (3, 32, 32))

    # dataloaders
    train_loader = dataloaders["train_loader"]
    test_loader = dataloaders["test_loader"]
    val_loader = dataloaders["val_loader"]

    print("#> Techinque: {}".format("Giunchiglia" if old_method else "Our approach"))

    # instantiate the optimizer
    optimizer = get_adam_optimizer(net, learning_rate, weight_decay=weight_decay)

    # define the cost function
    cost_function = torch.nn.BCELoss()

    # Resume training or start a new experiment
    training_params, val_params, start_epoch = resume_training(
        resume, model_folder, net, optimizer
    )

    # log on wandb if and only if the module is loaded
    if set_wandb:
        wandb.watch(net)

    # for each epoch, train the network and then compute evaluation results
    for e in tqdm.tqdm(range(start_epoch, epochs), desc="Epochs"):
        train_loss, train_accuracy = training_step(
            net=net,
            train=dataloaders["train"],
            R=dataloaders["train_R"],
            train_loader=iter(train_loader),
            optimizer=optimizer,
            cost_function=cost_function,
            title="Training",
            device=device,
        )

        # save the values in the metrics
        metrics["loss"]["train"] = train_loss
        metrics["acc"]["train"] = train_accuracy

        # validation set
        val_loss, val_accuracy, val_score = test_step(
            net=net,
            test_loader=iter(val_loader),
            cost_function=cost_function,
            title="Validation",
            test=dataloaders["val"],
            device=device,
        )

        # save the values in the metrics
        metrics["loss"]["val"] = val_loss
        metrics["acc"]["val"] = val_accuracy
        metrics["score"]["val"] = train_accuracy

        # save model and checkpoint
        training_params["start_epoch"] = e + 1  # epoch where to start

        # check if I have outperformed the best loss in the validation set
        if val_params["best_score"] < metrics["score"]["val"]:
            val_params["best_score"] = metrics["score"]["val"]
            # save best weights
            if not dry:
                torch.save(net.state_dict(), os.path.join(model_folder, "best.pth"))
        # what to save
        save_dict = {
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_params": training_params,
            "val_params": val_params,
        }
        # save current weights
        if not dry:
            torch.save(net.state_dict(), os.path.join(model_folder, "net.pth"))
            # save current settings
            torch.save(save_dict, os.path.join(model_folder, "ckpt.pth"))
            if e % save_every_epochs == 0:
                # Dump every checkpoint
                torch.save(
                    save_dict,
                    os.path.join(model_folder, "ckpt_e{}.pth".format(e + 1)),
                )
        del save_dict

        # logs to TensorBoard
        log_values(writer, e, train_loss, train_accuracy, "Train")
        log_values(writer, e, val_loss, val_accuracy, "Validation")
        writer.add_scalar("Learning rate", get_lr(optimizer), e)

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "train/train_loss": train_loss,
                    "train/train_accuracy": train_accuracy,
                    "val/val_loss": val_loss,
                    "val/val_accuracy": val_accuracy,
                    "val/val_score": val_score,
                    "learning_rate": get_lr(optimizer),
                }
            )

        # test value
        print("\nEpoch: {:d}".format(e + 1))
        print(
            "\t Training loss {:.5f}, Training accuracy {:.2f}%".format(
                train_loss, train_accuracy
            )
        )
        print(
            "\t Validation loss {:.5f}, Validation accuracy {:.2f}%, Validation score {:.2f}%".format(
                val_loss, val_accuracy, val_score
            )
        )
        print("-----------------------------------------------------")

    # compute final evaluation results
    print("#> After training:")

    # Test on best weights
    load_best_weights(net, model_folder)

    # test set
    test_loss, test_accuracy, test_score = test_step(
        net=net,
        test_loader=iter(test_loader),
        cost_function=cost_function,
        title="Test",
        test=dataloaders["test"],
        device=device,
    )

    # log values
    log_values(writer, epochs, test_loss, test_accuracy, "Test")

    # log on wandb if and only if the module is loaded
    if set_wandb:
        wandb.log(
            {
                "test/test_loss": test_loss,
                "test/test_accuracy": test_accuracy,
            }
        )

    print(
        "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test score {:.2f}%".format(
            test_loss, test_accuracy, test_score
        )
    )
    print("-----------------------------------------------------")

    print("#> Explanations")

    print("Get random example from test_loader...")

    test_el, _ = next(iter(test_loader))
    # move everything on the cpu
    net = net.to('cpu')
    net.R = net.R.to('cpu')
    # get the single element batch
    single_el = torch.unsqueeze(test_el[0], 0)
    # set the gradients as required
    single_el.requires_grad = True
    # get the predictions
    preds = net(single_el.float())

    grd = output_gradients(single_el, preds)[0]

    print("Gradient with respect to the input: {}".format(grd))
    if not old_method:
        # permute to show
        grd = grd.permute(1, 2, 0)
        grd = grd.numpy()
        # normalize
        grd = (grd - np.min(grd)) / (np.max(grd) - np.min(grd))
        fig = plt.figure()
        plt.imshow(grd)
        plt.title("Gradient with respect to the input")
        fig.savefig("{}/gradients.png".format(os.environ["IMAGE_FOLDER"]), dpi=fig.dpi)

    i_gradient, mean_grad = compute_integrated_gradient(
        test_el.float(), torch.zeros_like(single_el).float(), net
    )
    i_gradient = i_gradient[0]
    mean_grad = mean_grad[0]
    if not old_method:
        # permute to show
        i_gradient = i_gradient.permute(1, 2, 0)
        i_gradient = i_gradient.numpy()
        fig = plt.figure()
        i_gradient = (i_gradient - np.min(i_gradient)) / (
            np.max(i_gradient) - np.min(i_gradient)
        )
        plt.imshow(i_gradient)
        plt.title("Integrated Gradient with respect to the input")
        fig.savefig(
            "{}/integrated_gradients.png".format(os.environ["IMAGE_FOLDER"]),
            dpi=fig.dpi,
        )
    print(
        "Integrated gradient with respect to the input: {}\nMean Gradient {}".format(
            i_gradient, mean_grad
        )
    )

    # closes the logger
    writer.close()

    # set the dataset in case the dataset is not set
    if not old_method:
        dataset = "chmncc"

    f = open("results/" + dataset + ".csv", "a")
    f.write(str(kwargs.pop("seed")) + "," + str(epochs) + "," + str(test_score) + "\n")
    f.close()


def experiment(args: Namespace) -> None:
    """Experiment wrapper function

    Args:
      args (Namespace): command line arguments
    """

    print("\n### Experiment ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    if args.giunchiglia:
        assert args.dataset is not None

    # set wandb if needed
    if args.wandb:
        # Log in to your W&B account
        wandb.login()
        # set the argument to true
        args.set_wandb = args.wandb

    if args.wandb:
        # start the log
        wandb.init(project=args.project, entity=args.entity)

    # set the seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run the experiment
    c_hmcnn(**vars(args))

    # close wandb
    if args.wandb:
        # finish the log
        wandb.finish()


def main(args: Namespace) -> None:
    """Main function

    It runs the `func` function passed to the parser with the respective
    parameters
    Args:
      args (Namespace): command line arguments
    """

    # execute the function `func` with args as arguments
    args.func(
        args,
    )


if __name__ == "__main__":
    """
    Main

    Calls the main function with the command line arguments passed as parameters
    """
    main(get_args())
