"""Main module of the `c-hmcnn` project

This code has been developed by Eleonora Giunchiglia and Thomas Lukasiewicz; later modified by Samuele Bortolotti
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import torch
import tqdm
import cv2 as cv
import torch.nn as nn
from argparse import Namespace
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from typing import Any, Dict, List
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCELoss
import wandb
import signal
import matplotlib.image
import torch.multiprocessing
from chmncc.loss import RRRLoss, IGRRRLoss

torch.multiprocessing.set_sharing_strategy("file_system")

# data folder
os.environ["DATA_FOLDER"] = "./"
os.environ["MODELS"] = "./models"
os.environ["IMAGE_FOLDER"] = "./plots"

from chmncc.utils.utils import (
    force_prediction_from_batch,
    log_values,
    resume_training,
    get_lr,
    average_image_contributions,
    load_best_weights,
    load_last_weights,
    grouped_boxplot,
    plot_global_multiLabel_confusion_matrix,
    plot_confusion_matrix_statistics,
)
from chmncc.early_stopper import EarlyStopper
from chmncc.networks.ConstrainedFFNN import initializeConstrainedFFNNModel
from chmncc.networks import LeNet5, LeNet7, ResNet18, AlexNet, MLP
from chmncc.train import training_step
from chmncc.optimizers import (
    get_adam_optimizer,
    get_exponential_scheduler,
    get_sgd_optimizer,
    get_plateau_scheduler,
)
from chmncc.test import test_step, test_step_with_prediction_statistics
from chmncc.dataset import (
    load_old_dataloaders,
    load_cifar_dataloaders,
    get_named_label_predictions,
    LoadDebugDataset,
)
import chmncc.dataset.preproces_cifar as data
import chmncc.debug.debug as debug
import chmncc.dataset.visualize_dataset as visualize_data
from chmncc.config.old_config import lrs, epochss, hidden_dims
from chmncc.config import confunders, hierarchy
from chmncc.explanations import compute_integrated_gradient, output_gradients
from chmncc.revise import revise_step


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
    debug.configure_subparsers(subparsers)

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
        choices=["lenet", "lenet7", "resnet", "alexnet", "mlp"],
        default="lenet",
        help="CNN architecture",
    )
    parser.add_argument(
        "--pretrained", type=bool, default=False, help="load the pretrained model"
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
        "--confound",
        "-conf",
        dest="no_confounder",
        action="store_false",
        help="Use the Giunchiglia et al. layer to enforce hierarchical logical constraints",
    )
    parser.add_argument(
        "--no-confound",
        "-noconf",
        dest="no_confounder",
        action="store_true",
        help="Do not use the confounders in training and test",
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
        "--num-workers", type=int, default=4, help="dataloaders num workers"
    )
    parser.add_argument(
        "--prediction-treshold",
        type=float,
        default=0.5,
        help="considers the class to be predicted in a multilabel classification setting",
    )
    parser.add_argument("--patience", type=int, default=15, help="scheduler patience")
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
    # set the main function to run when blob is called from the command line
    parser.set_defaults(
        func=experiment,
        constrained_layer=True,
        no_confounder=False,
        force_prediction=False,
        fixed_confounder=False,
        use_softmax=False,
    )


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
    constrained_layer: bool = True,
    no_confounder: bool = False,
    force_prediction: bool = False,
    num_workers: int = 4,
    patience: int = 10,
    prediction_treshold: float = 0.5,
    fixed_confounder: bool = False,
    use_softmax: bool = False,
    **kwargs: Any,
) -> None:
    r"""
    Function which performs both training and test step

    Default:
        resume [bool] = False: by default do not resume last training
        device [str] = "cuda": move tensors on GPU, could be also "cpu"
        batch_size [int] = 128
        test_batch_size [int] = 128
        learning_rate [float] = 0.01
        epochs [int] = 30
        weight_decay [float] = 1e-5
        save_every_epochs [int] = 10: save a checkpoint every 10 epoch
        dry [bool] = False: by default save weights
        set_wandb [bool] = False,
        dataset [str] = "", taken for retrocompatibility with Giunchiglia et al approach
        network [str] = "lenet"
        pretrained [bool] = False
        constrained_layer [bool] = True
        no_confounder [bool] = False
        force_prediction [bool] = False
        num_workers [int] = 4
        patience [int] = 10
        prediction_treshold [float] = 0.01
        fixed_confounder [bool] = False
        use_softmax [bool] = False

    Args:
        exp_name [str]: name of the experiment, basically where to save the logs of the SummaryWriter
        resume [bool] = False: whether to resume a checkpoint
        device [str] = "cuda": where to load the tensors
        batch_size [int] = 128: default batch size
        test_batch_size [int] = 128: default batch size for the test set
        learning_rate [float] = 0.01: initial learning rate
        weight_decay [float] = 1e-5: weigt decay
        epochs [int] = 30: number of epochs
        save_every_epochs: int = 10: save a checkpoint every `save_every_epochs` epoch
        dry [bool] = False: whether to do not save weights
        set_wandb [bool] = False: whether to log values on wandb
        dataset [str] = str: dataset name: the dataset is specified -> old approach
        network [str] = "lenet": which arachitecture to employ
        pretrained [bool] = False, whether the network is pretrained [Note: lenet is not pretrained]
        constrained_layer [bool] = True: whether to use the constrained output layer from Giunchiglia et al.
        no_confounder [bool]: whether to have a normal, and therefore not confounded, training
        force_prediction [bool]: whether to force the prediction always
        num_workers [int]: number of workers for the dataloaders
        patience [int]: patience for the scheduler
        prediction_treshold [float]: prediction threshold
        fixed_confounder [bool] = False: use fixed confounder position
        use_softmax [bool] = False: whether to use softmax
        \*\*kwargs [Any]: additional key-value arguments
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
        "score": {"train": 0.0, "val": 0.0, "test": 0.0},
    }

    # get dataloaders
    if old_method:
        # Load the datasets
        dataloaders = load_old_dataloaders(dataset, batch_size, device=device)
    else:
        # img size for alexnet
        img_size = 32

        if network == "alexnet":
            img_size = 224

        dataloaders = load_cifar_dataloaders(
            img_size=img_size,  # the size is squared
            img_depth=3,  # number of channels
            device=device,
            csv_path="./dataset/train.csv",
            test_csv_path="./dataset/test_reduced.csv",
            val_csv_path="./dataset/val.csv",
            cifar_metadata="./dataset/pickle_files/meta",
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            normalize=True,  # normalize the dataset
            num_workers=num_workers,  # num workers
            fixed_confounder=fixed_confounder,
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
                dataloaders["train_R"],
                121,
                constrained_layer,
            )  # 20 superclasses, 100 subclasses + the root
        elif network == "lenet7":
            net = LeNet7(
                dataloaders["train_R"],
                121,
                constrained_layer,
                dataloaders["train_set"].n_superclasses,
                use_softmax,
            )  # 20 superclasses, 100 subclasses + the root
        elif network == "alexnet":
            # AlexNet
            net = AlexNet(
                dataloaders["train_R"], 121, constrained_layer
            )  # 20 superclasses, 100 subclasses + the root
        elif network == "mlp":
            # MLP
            net = MLP(
                dataloaders["train_R"],
                121,
                constrained_layer,
                dataloaders["train_set"].n_superclasses,
                use_softmax,
            )  # 20 superclasses, 100 subclasses + the root
        else:
            net = ResNet18(
                dataloaders["train_R"], 121, pretrained, constrained_layer
            )  # 20 superclasses, 100 subclasses + the root

    # move the network
    net = net.to(device)

    print("#> Model")

    # adjust image size
    img_size = 32
    if network == "alexnet":
        img_size = 224

    summary(net, (3, img_size, img_size))

    # dataloaders
    train_loader = dataloaders["train_loader"]

    test_loader = dataloaders["test_loader"]
    val_loader = dataloaders["val_loader"]

    confounder_mask_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position"],
    )
    confounder_mask_test = LoadDebugDataset(
        dataloaders["test_dataset_with_labels_and_confunders_pos"],
    )
    # Dataloaders for the previous values
    debug_loader = torch.utils.data.DataLoader(
        confounder_mask_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_debug = torch.utils.data.DataLoader(
        confounder_mask_test,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if no_confounder:
        print("I am not going to use the confounders!")
        train_loader = dataloaders["train_loader_no_confounder"]
        test_loader = dataloaders["test_loader_no_confounder"]
        val_loader = dataloaders["val_loader_no_confound"]

    print("#> Techinque: {}".format("Giunchiglia" if old_method else "Our approach"))

    # instantiate the optimizer
    optimizer = get_adam_optimizer(net, learning_rate, weight_decay=weight_decay)

    # scheduler
    scheduler = get_plateau_scheduler(optimizer=optimizer, patience=patience)

    # define the cost function
    cost_function = torch.nn.BCELoss()

    # EarlyStopper
    early_stopper = EarlyStopper(patience=3, min_delta=5)

    # Resume training or start a new experiment
    training_params, val_params, start_epoch = resume_training(
        resume, model_folder, net, optimizer
    )

    # log on wandb if and only if the module is loaded
    if set_wandb:
        wandb.watch(net)

    # for each epoch, train the network and then compute evaluation results
    for e in tqdm.tqdm(range(start_epoch, epochs), desc="Epochs"):
        (
            train_loss,
            train_accuracy,
            train_au_prc_score_raw,
            train_au_prc_score_const,
            train_loss_parent,
            train_loss_children,
        ) = training_step(
            net=net,
            train=dataloaders["train"],
            R=dataloaders["train_R"],
            train_loader=iter(train_loader),
            optimizer=optimizer,
            cost_function=cost_function,
            title="Training",
            device=device,
            constrained_layer=constrained_layer,
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            superclasses_number=dataloaders["train_set"].n_superclasses,
        )

        # save the values in the metrics
        metrics["loss"]["train"] = train_loss
        metrics["acc"]["train"] = train_accuracy
        metrics["score"]["train"] = train_au_prc_score_const

        # mettere qua il revise step
        (
            revise_total_loss,
            revise_total_right_answer_loss,
            revise_total_right_reason_loss,
            revise_total_accuracy,
            revise_total_score_raw,
            revise_total_score_const,
            revise_right_reason_loss_confounded,
        ) = revise_step(
            epoch_number=e,
            net=net,
            debug_loader=iter(debug_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=RRRLoss(
                net=net,
                regularizer_rate=1,
                base_criterion=BCELoss(),
            ),
            device=device,
            title="Train with RRR",
            gradient_analysis=False,
            have_to_train=False,
            folder_where_to_save=os.environ["IMAGE_FOLDER"],
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
        )

        # validation set
        val_loss, val_accuracy, val_score_raw, val_score_const, _, _ = test_step(
            net=net,
            test_loader=iter(val_loader),
            cost_function=cost_function,
            title="Validation",
            test=dataloaders["val"],
            device=device,
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            superclasses_number=dataloaders["train_set"].n_superclasses,
        )

        # save the values in the metrics
        metrics["loss"]["val"] = val_loss
        metrics["acc"]["val"] = val_accuracy
        metrics["score"]["val"] = train_accuracy

        # test values
        test_loss, test_accuracy, test_score_raw, test_score_const, _, _ = test_step(
            net=net,
            test_loader=iter(test_loader),
            cost_function=cost_function,
            title="Test",
            test=dataloaders["test"],
            device=device,
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            superclasses_number=dataloaders["train_set"].n_superclasses,
        )

        (
            test_revise_total_loss,
            test_revise_total_right_answer_loss,
            test_revise_total_right_reason_loss,
            test_revise_total_accuracy,
            test_revise_total_score_raw,
            test_revise_total_score_const,
            test_revise_right_reason_loss_confounded,
        ) = revise_step(
            epoch_number=e,
            net=net,
            debug_loader=iter(test_debug),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=RRRLoss(
                net=net,
                regularizer_rate=1,
                base_criterion=BCELoss(),
            ),
            device=device,
            title="Train with RRR",
            gradient_analysis=False,
            have_to_train=False,
            folder_where_to_save=os.environ["IMAGE_FOLDER"],
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
        )

        # save the values in the metrics
        metrics["loss"]["test"] = test_loss
        metrics["acc"]["test"] = test_accuracy
        metrics["score"]["test"] = test_score_const

        # save model and checkpoint
        training_params["start_epoch"] = e + 1  # epoch where to start

        # check if I have outperformed the best loss in the validation set
        if val_params["best_score"] < metrics["score"]["test"]:
            val_params["best_score"] = metrics["score"]["test"]
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
        log_values(writer, e, test_loss, test_accuracy, "Test")
        log_values(writer, e, train_loss, train_accuracy, "Train")
        log_values(writer, e, val_loss, val_accuracy, "Validation")
        print("Learning rate:", get_lr(optimizer))
        writer.add_scalar("Learning rate", get_lr(optimizer), e)

        # test value
        print("\nEpoch: {:d}".format(e + 1))
        print(
            "\t Training loss {:.5f}, Training accuracy {:.2f}%, Training Area under Precision-Recall Curve Raw {:.3f}, Training Area under Precision-Recall Curve Const {:.3f}".format(
                train_loss,
                train_accuracy,
                train_au_prc_score_raw,
                train_au_prc_score_const,
            )
        )
        print(revise_total_right_answer_loss, revise_total_right_reason_loss)
        print(
            "\t Validation loss {:.5f}, Validation accuracy {:.2f}%, Validation Area under Precision-Recall Curve Raw {:.3f},  Validation Area under Precision-Recall Curve Const {:.3f}".format(
                val_loss, val_accuracy, val_score_raw, val_score_const
            )
        )
        print(test_revise_total_right_answer_loss, test_revise_total_right_reason_loss)

        # log on wandb if and only if the module is loaded
        logs = {
            "train/train_loss": train_loss,
            "train/train_right_anwer_loss": train_loss,
            "train/train_accuracy": train_accuracy,
            "train/train_auprc_raw": train_au_prc_score_raw,
            "train/train_auprc_const": train_au_prc_score_const,
            "test/train_right_reason": revise_total_right_reason_loss,
            "val/val_loss": val_loss,
            "val/val_accuracy": val_accuracy,
            "val/val_auprc_raw": val_score_raw,
            "val/val_auprc_const": val_score_const,
            "learning_rate": get_lr(optimizer),
            "test/test_loss": test_loss,
            "test/test_right_answer_loss": test_loss,
            "test/test_accuracy": test_accuracy,
            "test/test_auprc_raw": test_score_raw,
            "test/test_auprc_const": test_score_const,
            "test/test_right_reason": test_revise_total_right_reason_loss,
        }

        if train_loss_parent is not None and train_loss_children is not None:
            logs.update({"train/train_right_answer_loss_parent": train_loss_parent})
            logs.update({"train/train_right_answer_loss_children": train_loss_children})

        if set_wandb:
            wandb.log(logs)

        print(
            "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve Raw {:.3f}, Test Area under Precision-Recall Curve Const {:.3f}".format(
                test_loss, test_accuracy, test_score_raw, test_score_const
            )
        )

        print("-----------------------------------------------------")

        # update scheduler
        scheduler.step(val_loss)

        # early stopping
        if early_stopper.early_stop(val_loss):
            print("Early Stopping!\n")
            break

    # compute final evaluation results
    print("#> After training:")

    # Test on best weights
    load_best_weights(net, model_folder, device)

    # test set
    test_loss, test_accuracy, test_score_raw, test_score_const, _, _ = test_step(
        net=net,
        test_loader=iter(test_loader),
        cost_function=cost_function,
        title="Test",
        test=dataloaders["test"],
        device=device,
        prediction_treshold=prediction_treshold,
        force_prediction=force_prediction,
        use_softmax=use_softmax,
        superclasses_number=dataloaders["train_set"].n_superclasses,
    )

    # log values
    log_values(writer, epochs, test_loss, test_accuracy, "Test")

    print(
        "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve Raw {:.3f}, Test Area under Precision-Recall Curve Const {:.3f}".format(
            test_loss, test_accuracy, test_score_raw, test_score_const
        )
    )
    print("-----------------------------------------------------")

    print("#> Explanations")

    print("Get random example from test_loader...")

    if old_method:
        test_el, _ = next(iter(test_loader))
    else:
        torch.save(net.state_dict(), os.path.join(model_folder, "last.pth"))

        # load the human readable labels dataloader
        test_loader_with_label_names = dataloaders["test_loader_with_labels_name"]
        test_dataset_with_label_names = dataloaders["test_set"]
        # load the training dataloader
        training_loader_with_labels_names = dataloaders[
            "training_loader_with_labels_names"
        ]

        labels_name = test_dataset_with_label_names.nodes_names_without_root

        ## TRAIN ##
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
            _
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(training_loader_with_labels_names),
            cost_function=cost_function,
            title="Collect Statistics [TRAIN]",
            test=dataloaders["train"],
            device=device,
            labels_name=labels_name,
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            superclasses_number=dataloaders["train_set"].n_superclasses,
        )

        ## ! Confusion matrix !
        plot_global_multiLabel_confusion_matrix(
            y_test=y_test,
            y_est=y_pred,
            label_names=labels_name,
            size=(30, 30),
            fig_name="{}/train_confusion_matrix_normalized".format(
                os.environ["IMAGE_FOLDER"]
            ),
            normalize=True,
        )
        plot_global_multiLabel_confusion_matrix(
            y_test=y_test,
            y_est=y_pred,
            label_names=labels_name,
            size=(30, 30),
            fig_name="{}/train_confusion_matrix".format(os.environ["IMAGE_FOLDER"]),
            normalize=False,
        )
        plot_confusion_matrix_statistics(
            clf_report=clf_report,
            fig_name="{}/train_confusion_matrix_statistics.png".format(
                os.environ["IMAGE_FOLDER"]
            ),
        )
        # grouped boxplot
        grouped_boxplot(
            statistics_predicted,
            os.environ["IMAGE_FOLDER"],
            "Predicted",
            "Not predicted",
            "train_predicted",
        )
        grouped_boxplot(
            statistics_correct,
            os.environ["IMAGE_FOLDER"],
            "Correct prediction",
            "Wrong prediction",
            "train_accuracy",
        )

        ## TEST ##
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
            _
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(test_loader_with_label_names),
            cost_function=cost_function,
            title="Collect Statistics [TEST]",
            test=dataloaders["test"],
            device=device,
            labels_name=labels_name,
            prediction_treshold=prediction_treshold,
            force_prediction=force_prediction,
            use_softmax=use_softmax,
            superclasses_number=dataloaders["train_set"].n_superclasses,
        )

        ## ! Confusion matrix !
        plot_global_multiLabel_confusion_matrix(
            y_test=y_test,
            y_est=y_pred,
            label_names=labels_name,
            size=(30, 30),
            fig_name="{}/test_confusion_matrix_normalized".format(
                os.environ["IMAGE_FOLDER"]
            ),
            normalize=True,
        )
        plot_global_multiLabel_confusion_matrix(
            y_test=y_test,
            y_est=y_pred,
            label_names=labels_name,
            size=(30, 30),
            fig_name="{}/test_confusion_matrix".format(os.environ["IMAGE_FOLDER"]),
            normalize=False,
        )
        plot_confusion_matrix_statistics(
            clf_report=clf_report,
            fig_name="{}/test_confusion_matrix_statistics.png".format(
                os.environ["IMAGE_FOLDER"]
            ),
        )
        # grouped boxplot
        grouped_boxplot(
            statistics_predicted,
            os.environ["IMAGE_FOLDER"],
            "Predicted",
            "Not predicted",
            "test_predicted",
        )
        grouped_boxplot(
            statistics_correct,
            os.environ["IMAGE_FOLDER"],
            "Correct prediction",
            "Wrong prediction",
            "test_accuracy",
        )
        # extract also the names of the classes
        test_el, superclass, subclass, _ = next(iter(test_loader_with_label_names))

    # move everything on the cpu
    net = net.to("cpu")
    net.R = net.R.to("cpu")

    # explainations
    for i in range(test_el.shape[0]):
        # whether the sample is confunded
        confunded = False
        # get the single element batch
        single_el = torch.unsqueeze(test_el[i], 0)
        # set the gradients as required
        single_el.requires_grad = True
        # get the predictions
        preds = net(single_el.float())
        # output gradients
        grd = output_gradients(single_el, preds)[0]
        # orginal image
        fig = plt.figure()
        # prepare for the show
        single_el_show = single_el[0].clone().cpu().data.numpy()
        single_el_show = single_el_show.transpose(1, 2, 0)
        # normalize
        single_el_show = np.fabs(single_el_show)
        single_el_show = single_el_show / np.max(single_el_show)
        plt.imshow(single_el_show)

        if old_method:
            plt.title("Random Sample")
        else:
            # get named predictions
            torch.set_printoptions(profile="full")
            # get the prediction
            if force_prediction:
                predicted_1_0 = force_prediction_from_batch(
                    preds.data,
                    prediction_treshold,
                    use_softmax,
                    dataloaders["train_set"].n_superclasses,
                )
            else:
                predicted_1_0 = preds.data > prediction_treshold

            predicted_1_0 = predicted_1_0.to(torch.float)[0]
            # get the named prediction
            named_prediction = get_named_label_predictions(
                predicted_1_0, dataloaders["test_set"].get_nodes()
            )
            # extract parent and children
            parents = hierarchy.keys()
            children = [
                element
                for element_list in hierarchy.values()
                for element in element_list
            ]
            parent_predictions = list(filter(lambda x: x in parents, named_prediction))
            children_predictions = list(
                filter(lambda x: x in children, named_prediction)
            )
            # select whether it is confunded
            print(superclass[i], subclass[i])
            if superclass[i] in confunders:
                for tmp_index in range(len(confunders[superclass[i]]["test"])):
                    if (
                        confunders[superclass[i]]["test"][tmp_index]["subclass"]
                        == subclass[i]
                    ):
                        print("Found confunder!")
                        confunded = True
                        break

            if not confunded:
                continue

            print(i, parent_predictions, children_predictions)

            # plot the title
            prediction_text = "Predicted: {}\nbecause of: {}".format(
                parent_predictions, children_predictions
            )
            plt.title(
                "Groundtruth superclass: {} \nGroundtruth subclass: {}\n\n{}".format(
                    superclass[i], subclass[i], prediction_text
                )
            )
            plt.tight_layout()

        fig.savefig(
            "{}/{}_{}_original{}.png".format(
                os.environ["IMAGE_FOLDER"],
                i,
                network,
                "_confunded" if confunded else "",
            ),
            dpi=fig.dpi,
        )
        plt.close()

        if not old_method:
            # permute to show
            grd = grd.permute(1, 2, 0)
            grd = grd.cpu().data.numpy()
            grd = average_image_contributions(grd)
            # normalize
            grd = np.fabs(grd)
            #  grd = grd / np.max(grd)
            fig = plt.figure()
            plt.imshow(grd, cmap="gray")
            plt.title("Gradient with respect to the input")
            # norm color
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(grd))
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="gray"),
                label="Gradient magnitude",
            )
            fig.savefig(
                "{}/{}_{}_gradients.png".format(os.environ["IMAGE_FOLDER"], i, network),
                dpi=fig.dpi,
            )
            plt.close()

            # overlayed image
            fig = plt.figure()
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
                label="Gradient magnitude overlayed",
            )
            plt.imshow(single_el_show)
            plt.imshow(grd, cmap="viridis", alpha=0.5)
            plt.title("Input gradients")

            # show the figure
            fig.savefig(
                "{}/{}_{}_gradients_overlayed.png".format(
                    os.environ["IMAGE_FOLDER"], i, network
                ),
                dpi=fig.dpi,
            )
            plt.close()

        i_gradient = compute_integrated_gradient(
            single_el, torch.zeros_like(single_el), net
        )

        if not old_method:
            # permute to show
            i_gradient = i_gradient.permute(1, 2, 0)
            i_gradient = i_gradient.cpu().data.numpy()
            i_gradient = average_image_contributions(i_gradient)
            # get the numpy array
            # get the absolute value
            i_gradient = np.fabs(i_gradient)
            # normalize the value
            #  i_gradient = i_gradient / np.max(i_gradient)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(i_gradient))
            # figure
            fig = plt.figure()
            # show
            plt.imshow(i_gradient, cmap="gray")
            plt.title("Integrated Gradient with respect to the input")
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="gray"),
                label="Gradient magnitude",
            )
            fig.savefig(
                "{}/{}_{}_integrated_gradients.png".format(
                    os.environ["IMAGE_FOLDER"], i, network
                ),
                dpi=fig.dpi,
            )
            plt.close()

            # overlayed image
            fig = plt.figure()
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
                label="Gradient magnitude overlayed",
            )
            plt.imshow(single_el_show)
            plt.imshow(i_gradient, cmap="viridis", alpha=0.5)
            plt.title("Integrated gradients")

            # show the figure
            fig.savefig(
                "{}/{}_{}_integrated_gradients_overlayed.png".format(
                    os.environ["IMAGE_FOLDER"], i, network
                ),
                dpi=fig.dpi,
            )
            plt.close()

    print("Done with the explainations")

    # closes the logger
    writer.close()

    # set the dataset in case the dataset is not set
    if not old_method:
        dataset = "chmncc"

    f = open("results/" + dataset + ".csv", "a")
    f.write(
        str(kwargs.pop("seed")) + "," + str(epochs) + "," + str(test_score_const) + "\n"
    )
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

    # disable tensorflow warnings
    main(get_args())
