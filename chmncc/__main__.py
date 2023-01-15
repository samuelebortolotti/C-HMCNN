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
import wandb
import signal
import matplotlib.image

# data folder
os.environ["DATA_FOLDER"] = "./"
os.environ["MODELS"] = "./models"
os.environ["IMAGE_FOLDER"] = "./plots"

from chmncc.utils.utils import (
    log_values,
    resume_training,
    get_lr,
    average_image_contributions,
    load_best_weights,
)
from chmncc.networks.ConstrainedFFNN import initializeConstrainedFFNNModel
from chmncc.networks import LeNet5, ResNet18, AlexNet
from chmncc.train import training_step
from chmncc.optimizers import get_adam_optimizer
from chmncc.test import test_step, test_step_with_prediction_statistics
from chmncc.dataset import (
    load_old_dataloaders,
    load_cifar_dataloaders,
    get_named_label_predictions,
)
import chmncc.dataset.preproces_cifar as data
import chmncc.debug.debug as debug
import chmncc.dataset.visualize_dataset as visualize_data
from chmncc.config.old_config import lrs, epochss, hidden_dims
from chmncc.config import confunders, hierarchy
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
        choices=["lenet", "resnet", "alexnet"],
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
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=experiment, constrained_layer=True)


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
                dataloaders["train_R"], 121, constrained_layer
            )  # 20 superclasses, 100 subclasses + the root
        elif network == "alexnet":
            # AlexNet
            net = AlexNet(
                dataloaders["train_R"], 121, constrained_layer
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
        train_loss, train_accuracy, train_au_prc_score = training_step(
            net=net,
            train=dataloaders["train"],
            R=dataloaders["train_R"],
            train_loader=iter(train_loader),
            optimizer=optimizer,
            cost_function=cost_function,
            title="Training",
            device=device,
            constrained_layer=constrained_layer,
        )

        # save the values in the metrics
        metrics["loss"]["train"] = train_loss
        metrics["acc"]["train"] = train_accuracy
        metrics["score"]["train"] = train_au_prc_score

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
                    "train/train_auprc": train_au_prc_score,
                    "val/val_loss": val_loss,
                    "val/val_accuracy": val_accuracy,
                    "val/val_auprc": val_score,
                    "learning_rate": get_lr(optimizer),
                }
            )

        # test value
        print("\nEpoch: {:d}".format(e + 1))
        print(
            "\t Training loss {:.5f}, Training accuracy {:.2f}%, Training Area under Precision-Recall Curve {:.3f}".format(
                train_loss, train_accuracy, train_au_prc_score
            )
        )
        print(
            "\t Validation loss {:.5f}, Validation accuracy {:.2f}%, Validation Area under Precision-Recall Curve {:.3f}".format(
                val_loss, val_accuracy, val_score
            )
        )
        print("-----------------------------------------------------")

    # compute final evaluation results
    print("#> After training:")

    # Test on best weights
    load_best_weights(net, model_folder, device)

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
                "test/test_auprc": test_score,
            }
        )

    print(
        "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
            test_loss, test_accuracy, test_score
        )
    )
    print("-----------------------------------------------------")

    print("#> Explanations")

    print("Get random example from test_loader...")

    if old_method:
        test_el, _ = next(iter(test_loader))
    else:
        # load the human readable labels dataloader
        test_loader_with_label_names = dataloaders["test_loader_with_labels_name"]
        # extract also the names of the classes
        test_el, superclass, subclass, _ = next(iter(test_loader_with_label_names))
        # collect stats
        (
            _,
            _,
            _,
            statistics_predicted,
            statistics_correct,
        ) = test_step_with_prediction_statistics(
            net=net,
            test_loader=iter(test_loader_with_label_names),
            cost_function=cost_function,
            title="Collect Statistics",
            test=dataloaders["test"],
            device=device,
        )
        # grouped boxplot
        grouped_boxplot(
            statistics_predicted,
            os.environ["IMAGE_FOLDER"],
            "Predicted",
            "Not predicted",
            "predicted",
        )
        grouped_boxplot(
            statistics_correct,
            os.environ["IMAGE_FOLDER"],
            "Correct prediction",
            "Wrong prediction",
            "accuracy",
        )

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
            predicted_1_0 = preds.data > 0.5
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

        #  print("Gradient with respect to the input: {}".format(grd))
        if not old_method:
            # permute to show
            grd = grd.permute(1, 2, 0)
            grd = grd.cpu().data.numpy()
            grd = average_image_contributions(grd)
            # normalize
            grd = np.fabs(grd)
            grd = grd / np.max(grd)
            fig = plt.figure()
            plt.imshow(grd, cmap="gray")
            plt.title("Gradient with respect to the input")
            fig.savefig(
                "{}/{}_{}_gradients.png".format(os.environ["IMAGE_FOLDER"], i, network),
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
            i_gradient = i_gradient / np.max(i_gradient)
            # save the raw image
            matplotlib.image.imsave(
                "{}/{}_{}_integrated_gradients_raw.png".format(
                    os.environ["IMAGE_FOLDER"], i, network
                ),
                i_gradient,
            )
            # figure
            fig = plt.figure()
            # show
            plt.imshow(i_gradient, cmap="gray")
            plt.title("Integrated Gradient with respect to the input")
            fig.savefig(
                "{}/{}_{}_integrated_gradients.png".format(
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run the experiment
    c_hmcnn(**vars(args))

    # close wandb
    if args.wandb:
        # finish the log
        wandb.finish()


def split(a: List[Any], n: int) -> List[Any]:
    """Split an array into equal intervals

    Args:
      a [List[Any]]: list
      n [int]: number of equal intervals

    Returns:
      list separaed with equal intervals
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def grouped_boxplot(
    statistics: Dict[str, List[int]],
    image_folder: str,
    correct_txt: str,
    wrong_txt: str,
    statistics_name: str,
) -> None:
    """Grouped Boxplot
    print the grouped boxplot for the statistics

    Args:
      statistics [Dict[List[int]]]: set of statistics
      image_folder [str]: image folde
    """
    predicted = []
    unpredicted = []
    index = []
    # create the statistics
    for key, item in statistics.items():
        if key != "total":
            predicted.append(item[1])
            unpredicted.append(item[0])
            index.append(key)

    fig = plt.figure(figsize=(8, 4))
    titles = np.array([correct_txt, wrong_txt])
    values = np.array([statistics["total"][1], statistics["total"][0]])
    plot = pd.Series(values).plot(kind="bar", color=["green", "red"])
    plot.bar_label(plot.containers[0], label_type="edge")
    plot.set_xticklabels(titles)
    plt.xticks(rotation=0)
    plt.title("Total: {} vs {}".format(correct_txt, wrong_txt))
    plt.tight_layout()
    fig.savefig("{}/statistics_{}_total.png".format(image_folder, statistics_name))
    plt.close()

    # print data
    for i, (el_p, el_u, el_i) in enumerate(
        zip(split(predicted, 10), split(unpredicted, 10), split(index, 10))
    ):
        # data
        data = {correct_txt: el_p, wrong_txt: el_u}
        # figure
        df = pd.DataFrame(data, index=el_i)
        plot = df.plot.bar(rot=0, figsize=(11, 9), color=["green", "red"])
        plot.bar_label(plot.containers[0], label_type="edge")
        plot.bar_label(plot.containers[1], label_type="edge")
        plt.title("{} vs {}".format(correct_txt, wrong_txt))
        plt.xticks(rotation=60)
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()
        fig = plot.get_figure()
        fig.savefig("{}/statistics_{}_{}.png".format(image_folder, statistics_name, i))
        plt.close()


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
