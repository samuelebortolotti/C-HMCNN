from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from numpy.lib.function_base import iterable
import torch
import os
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from chmncc.dataset.load_cifar import LoadDataset
from chmncc.networks import ResNet18, LeNet5
from chmncc.config import hierarchy
from chmncc.utils.utils import load_best_weights
from chmncc.dataset import (
    load_cifar_dataloaders,
    get_named_label_predictions,
)
import wandb
from chmncc.test import test_step
from chmncc.explanations import compute_integrated_gradient, output_gradients
from chmncc.loss import RRRLoss, IGRRRLoss
from chmncc.revise import revise_step
from chmncc.optimizers import get_adam_optimizer
from typing import Dict, Any, Tuple, Union, List
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchsummary import summary
from torch.utils.data import Dataset


class LoadDebugDataset(Dataset):
    """Loads the data from a pre-existing DataLoader (it should return the position of the confounder as well as the labels)"""

    def __init__(
        self,
        train_set: LoadDataset,
        device: str,
    ):
        """Init param"""
        self.train_set = train_set
        self.device = device

    def __len__(self) -> int:
        """Returns the total amount of data.
        Returns:
            number of dataset entries [int]
        """
        return len(self.train_set)

    def __getitem__(self, idx: int):
        """Returns the data"""
        (
            train_sample,
            _,
            _,
            hierarchical_label,
            confunder_pos1_x,
            confunder_pos1_y,
            confunder_pos2_x,
            confunder_pos2_y,
            confunder_shape,
        ) = self.train_set[idx]

        # parepare the train example and the explainations in the right shape
        single_el = prepare_single_test_sample(
            sample_batch=train_sample, device=self.device
        )

        # debug iteration to get the counfounder mask, without printing the image ofc
        confounder_mask, confounded = compute_mask(
            shape=(train_sample.shape[1], train_sample.shape[2]),
            confunder_pos1_x=confunder_pos1_x,
            confunder_pos1_y=confunder_pos1_y,
            confunder_pos2_x=confunder_pos2_x,
            confunder_pos2_y=confunder_pos2_y,
            confunder_shape=confunder_shape,
        )
        single_el.requires_grad = False
        # returns
        return (single_el, hierarchical_label, confounder_mask, confounded)


def prepare_single_test_sample(sample_batch: torch.Tensor, device: str) -> torch.Tensor:
    """Extract one element from the batch and sets gradients together with an additive element in the
    size in order to make it compliant with the rest of the algorithm.
    Args:
        sample_batch [torch.Tensor]: batch of Test samples
        device [str]: device
    Returns:
        element [torch.Tensor]: single element
    """
    # get the single element batch
    single_el = sample_batch
    # requires grad
    single_el.requires_grad = True
    # move the element to device
    single_el = single_el.to(device)
    # returns element and predictions
    return single_el


def save_test_sample(
    test_sample: torch.Tensor,
    prediction: torch.Tensor,
    idx: int,
    debug_folder: str,
    dataloaders: Dict[str, Any],
    superclass: str,
    subclass: str,
    integrated_gradients: bool,
    iteration: int,
    dry: bool,
) -> bool:
    """Save the test sample and tell whether the machine got the prediction right.
    Then it returns whether the sample contains a confunder or if the sample has been
    guessed correctly.

    Args:
        test_sample [torch.Tensor]: test sample depicting the image
        prediction [torch.Tensor]: prediction of the network on top of the test function
        idx [int]: index of the element of the batch to consider
        debug_folder [str]: string depicting the folder where to save the figures
        dataloaders [Dict[str, Any]]: dictionary depicting the dataloaders of the data
        superclass [str]: superclass string
        subclass [str]: subclass string
        integrated_gradients [bool]: whether to use integrated gradients
        iteration [int]: debug iteration
        dry [bool]: whether to save the image

    Returns:
        correct_guess [bool]: whether the model has guessed correctly
    """
    # prepare the element in order to show it
    single_el_show = test_sample[0].clone().cpu().data.numpy()
    single_el_show = single_el_show.transpose(1, 2, 0)

    # normalize
    single_el_show = np.fabs(single_el_show)
    single_el_show = single_el_show / np.max(single_el_show)

    # get the prediction
    predicted_1_0 = prediction.cpu().data > 0.5
    predicted_1_0 = predicted_1_0.to(torch.float)[0]

    # get the named prediction
    named_prediction = get_named_label_predictions(
        predicted_1_0, dataloaders["test_set"].get_nodes()
    )

    # extract parent and children prediction
    parents = hierarchy.keys()
    children = [
        element for element_list in hierarchy.values() for element in element_list
    ]
    parent_predictions = list(filter(lambda x: x in parents, named_prediction))
    children_predictions = list(filter(lambda x: x in children, named_prediction))

    # check whether it is confunded
    correct_guess = False

    # set the guess as correct if it is
    if len(parent_predictions) == 1 and len(children_predictions) == 1:
        if (
            superclass.strip() == parent_predictions[0].strip()
            and subclass == children_predictions[0].strip()
        ):
            correct_guess = True

    # get figure
    if not dry:
        # plot the title
        prediction_text = "Predicted: {}\nbecause of: {}".format(
            parent_predictions, children_predictions
        )

        fig = plt.figure()
        plt.imshow(single_el_show)
        plt.title(
            "Groundtruth superclass: {} \nGroundtruth subclass: {}\n\n{}".format(
                superclass, subclass, prediction_text
            )
        )
        plt.tight_layout()
        # show the figure
        fig.savefig(
            "{}/iter_{}_{}_original_{}{}.png".format(
                debug_folder,
                iteration,
                idx,
                "integrated_gradients" if integrated_gradients else "",
                "_correct" if correct_guess else "",
            ),
            dpi=fig.dpi,
        )
        # close the figure
        plt.close()

    # whether the prediction was right
    return correct_guess


def compute_gradients(
    single_el: torch.Tensor,
    net: nn.Module,
    integrated_gradients: bool,
    dry: bool,
    debug_folder: str,
    iteration: int,
    idx: int,
    correct_guess: bool = False,
) -> torch.Tensor:
    """Computes the integrated graditents or input gradient and optionally saves them in an image

    Args:
        single_el [torch.Tensor]: sample depicting the image
        net [nn.Module]: neural network
        integrated_gradients [bool]: whether the method uses integrated gradients or input gradients
        dry [bool]: whether to not save the images
        debug_folder [str]: string depicting the folder path
        iteration [int]: debug iteration
        idx [int]: index of the image
        correct_guess [bool]: whether it is a correct guess or not
    Returns:
        integrated_gradient [torch.Tensor]: integrated_gradient
    """
    if integrated_gradients:
        # integrated gradients
        gradient = compute_integrated_gradient(
            single_el, torch.zeros_like(single_el), net
        )
    else:
        # integrated gradients
        gradient = output_gradients(single_el, net(single_el))[0]

    # sum over RGB channels
    gradient = torch.sum(gradient, dim=0)

    # permute to show
    gradient_to_show = gradient.clone()
    # get the absolute value
    gradient_to_show = torch.abs(gradient_to_show)
    # normalize the value
    gradient_to_show = gradient_to_show / torch.max(gradient_to_show)

    if not dry:
        # show
        fig = plt.figure()
        plt.imshow(gradient_to_show.cpu().data.numpy(), cmap="gray")
        plt.title(
            "{} Gradient with respect to the input".format(
                "Integrated" if integrated_gradients else "Input"
            )
        )
        # show the figure
        fig.savefig(
            "{}/iter_{}_{}_gradient_{}{}.png".format(
                debug_folder,
                iteration,
                idx,
                "integrated" if integrated_gradients else "input",
                "_correct" if correct_guess else "",
            ),
            dpi=fig.dpi,
        )
        plt.close()
    return gradient


def compute_mask(
    shape: Tuple[int],
    confunder_pos1_x: int,
    confunder_pos1_y: int,
    confunder_pos2_x: int,
    confunder_pos2_y: int,
    confunder_shape: str,
) -> Tuple[torch.Tensor, bool]:
    """Debug iteration, basically a training with the RRR loss

    Args:
        debug_folder [str]: string depicting the folder path
        idx [int]: index of the image
        gradient [torch.Tensor]: integrated gradient tensor
        confuder_pos1_x [int]: x of the starting point
        confuder_pos1_y [int]: y of the starting point
        confuder_pos2_x [int]: x of the ending point
        confuder_pos2_y [int]: y of the ending point
        confunder_shape [Dict[str, Any]]: confunder information
        integrated_gradients [bool]: whether the method uses integrated gradients or input gradients
        correct_guess [bool]: whether the guess was correct
        iteration [int]: debug iteration

    Returns:
        confounder_mask [torch.Tensor]: tensor highlighting the area where the confounder is present with ones. It is zero elsewhere
    """
    # confounder mask
    confounder_mask = np.zeros(shape)
    # whether the example is confounded
    confounded = True

    if confunder_shape == "rectangle":
        # get the image of the modified gradient
        cv2.rectangle(
            confounder_mask,
            (confunder_pos1_x, confunder_pos1_y),
            (confunder_pos2_x, confunder_pos2_y),
            (255, 255, 255),
            cv2.FILLED,
        )
    elif confunder_shape == "circle":
        # get the image of the modified gradient
        cv2.circle(
            confounder_mask,
            (confunder_pos1_x, confunder_pos1_y),
            confunder_pos2_x,
            (255, 255, 255),
            cv2.FILLED,
        )
    else:
        confounded = False

    # binarize the mask and adjust it the right way
    confounder_mask = torch.tensor((confounder_mask > 0.5).astype(np.float_))

    # return the confounder mask
    return confounder_mask, confounded


def debug(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    iterations: int,
    cost_function: torch.nn.BCELoss,
    device: str,
    set_wandb: bool,
    integrated_gradients: bool,
    optimizer: torch.optim.Optimizer,
    debug_test_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    batch_size: int,
    reviseLoss: Union[RRRLoss, IGRRRLoss],
    **kwargs: Any
):
    """Method which performs the debug step, by detecting the confunded images first;
    then correcting the integrated gradient associated and finally re-training the model with the RRR loss

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
        title [str]: title of the tqdm
        batch_size [int]: size of the batch
        test_batch_size [int] size of the batch in the test settings
        debug_train_loader [torch.utils.data.DataLoader]: dataloader for the debug
        debug_test_loader [torch.utils.data.DataLoader]: dataloader for the test
        test_loader [torch.utils.data.DataLoader]: dataloader for final testing
        reviseLoss: Union[RRRLoss, IGRRRLoss]: loss for feeding the network some feedbacks
        **kwargs [Any]: kwargs
    """

    print("Have to run for {} debug iterations...".format(iterations))

    debug_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position"],
        device=device,
        integrated_gradients=integrated_gradients,
    )
    only_conf_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position_only_conf"],
        integrated_gradients=integrated_gradients,
        device=device,
    )
    no_conf_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position_no_conf"],
        integrated_gradients=integrated_gradients,
        device=device,
    )

    debug_loader = torch.utils.data.DataLoader(
        debug_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    debug_conf_loader = torch.utils.data.DataLoader(
        only_conf_train, batch_size=batch_size, shuffle=False, num_workers=4
    )
    debug_no_conf_loader = torch.utils.data.DataLoader(
        no_conf_train, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # running for the requested iterations
    for it in range(iterations):

        print("Start iteration number {}".format(it))
        print("-----------------------------------------------------")

        # training with RRRloss feedbacks
        (
            total_loss,
            total_right_answer_loss,
            total_right_reason_loss,
            total_accuracy,
            total_score,
        ) = revise_step(
            net=net,
            debug_loader=iter(debug_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=reviseLoss,
            device=device,
            title="Train with RRR",
        )

        print(
            "\n\t Debug full loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, Score {:.5f}".format(
                total_loss,
                total_right_answer_loss,
                total_right_reason_loss,
                total_accuracy,
                total_score,
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "debug/full_rrr_loss": total_loss,
                    "debug/full_right_anwer_loss": total_right_answer_loss,
                    "debug/full_right_reason_loss": total_right_reason_loss,
                    "debug/full_accuracy": total_accuracy,
                    "debug/full_score": total_score,
                }
            )

        (
            conf_loss,
            conf_right_answer_loss,
            conf_right_reason_loss,
            conf_accuracy,
            conf_score,
        ) = revise_step(
            net=net,
            debug_loader=iter(debug_conf_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=reviseLoss,
            device=device,
            title="Train with RRR [confounded]",
            have_to_train=False,
        )

        print(
            "\n\t Debug confounded samples loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, Score {:.5f}".format(
                conf_loss,
                conf_right_answer_loss,
                conf_right_reason_loss,
                conf_accuracy,
                conf_score,
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "debug/conf_rrr_loss": conf_loss,
                    "debug/conf_right_anwer_loss": conf_right_answer_loss,
                    "debug/conf_right_reason_loss": conf_right_reason_loss,
                    "debug/conf_accuracy": conf_accuracy,
                    "debug/conf_score": conf_score,
                }
            )

        (
            not_conf_loss,
            not_conf_right_answer_loss,
            not_conf_right_reason_loss,
            not_conf_accuracy,
            not_conf_score,
        ) = revise_step(
            net=net,
            debug_loader=iter(debug_no_conf_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=reviseLoss,
            device=device,
            title="Test with RRR [not confounded]",
            have_to_train=False,
        )

        print(
            "\n\t Debug not confounded samples loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, Score {:.5f}".format(
                not_conf_loss,
                not_conf_right_answer_loss,
                not_conf_right_reason_loss,
                not_conf_accuracy,
                not_conf_score,
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "debug/not_conf_rrr_loss": not_conf_loss,
                    "debug/not_conf_right_anwer_loss": not_conf_right_answer_loss,
                    "debug/not_conf_right_reason_loss": not_conf_right_reason_loss,
                    "debug/not_conf_accuracy": not_conf_accuracy,
                    "debug/not_conf_score": not_conf_score,
                }
            )

        print("Testing...")

        # validation set
        val_loss, val_accuracy, val_score = test_step(
            net=net,
            test_loader=iter(debug_test_loader),
            cost_function=cost_function,
            title="Validation",
            test=dataloaders["train"],
            device=device,
        )

        print(
            "\n\t [Validation set]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve {:.3f}".format(
                val_loss, val_accuracy, val_score
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                    "val/score": val_score,
                }
            )

        # validation set
        test_loss, test_accuracy, test_score = test_step(
            net=net,
            test_loader=iter(test_loader),
            cost_function=cost_function,
            title="Test",
            test=dataloaders["test"],
            device=device,
        )

        print(
            "\n\t [Test set]: Loss {:.5f}, Accuracy {:.2f}%, Area under Precision-Recall Curve {:.3f}".format(
                test_loss, test_accuracy, test_score
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "test/loss": val_loss,
                    "test/accuracy": val_accuracy,
                    "test/score": val_score,
                }
            )

        print("Done with debug for iteration number: {}".format(iterations))

        print("-----------------------------------------------------")


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the debug of the network
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("debug", help="Debug network subparser")
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["resnet", "lenet"],
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
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decy")
    parser.add_argument(
        "--integrated-gradients",
        "-igrad",
        dest="integrated_gradients",
        action="store_true",
        help="Use integrated gradients [default]",
    )
    parser.add_argument(
        "--no-integrated-gradients",
        "-noigrad",
        dest="integrated_gradients",
        action="store_false",
        help="Use input gradients",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Test batch size",
    )
    parser.add_argument(
        "--project", "-w", type=str, default="chmcnn-project", help="wandb project"
    )
    parser.add_argument(
        "--entity", "-e", type=str, default="samu32", help="wandb entity"
    )
    parser.add_argument("--wandb", "-wdb", type=bool, default=False, help="wandb")
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main, integrated_gradients=True)


def main(args: Namespace) -> None:
    """Checks the command line arguments and then runs the debug

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

    # Load dataloaders
    dataloaders = load_cifar_dataloaders(
        img_size=32,  # the size is squared
        img_depth=3,  # number of channels
        device=args.device,
        csv_path="./dataset/train.csv",
        test_csv_path="./dataset/test_reduced.csv",
        val_csv_path="./dataset/val.csv",
        cifar_metadata="./dataset/pickle_files/meta",
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        normalize=True,  # normalize the dataset
    )

    # Load dataloaders
    print("Load network weights...")

    # Network
    if args.network == "lenet":
        net = LeNet5(
            dataloaders["train_R"], 121
        )  # 20 superclasses, 100 subclasses + the root
    else:
        net = ResNet18(
            dataloaders["train_R"], 121, False
        )  # 20 superclasses, 100 subclasses + the root

    # move everything on the cpu
    net = net.to(args.device)
    net.R = net.R.to(args.device)
    # zero grad
    net.zero_grad()
    # show a summary
    summary(net, (3, 32, 32))

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

    optimizer = get_adam_optimizer(
        net, args.learning_rate, weight_decay=args.weight_decay
    )

    # Test on best weights
    load_best_weights(net, args.weights_path_folder, args.device)

    # dataloaders
    train_loader = dataloaders["train_loader_debug_mode"]
    test_loader = dataloaders["test_loader"]
    val_loader = dataloaders["val_loader"]

    # define the cost function
    cost_function = torch.nn.BCELoss()

    # test set
    test_loss, test_accuracy, test_score = test_step(
        net=net,
        test_loader=iter(test_loader),
        cost_function=cost_function,
        title="Test",
        test=dataloaders["test"],
        device=args.device,
    )

    print("Network resumed, performances:")

    print(
        "\n\t [TEST SET]: Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
            test_loss, test_accuracy, test_score
        )
    )

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.log(
            {
                "test/pre_exp_loss": test_loss,
                "test/pre_exp_accuracy": test_accuracy,
                "test/pre_exp_score": test_score,
            }
        )

    print("-----------------------------------------------------")

    print("#> Debug...")

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.watch(net)

    if not args.integrated_gradients:
        reviseLoss = RRRLoss(net=net, regularizer_rate=20, base_criterion=BCELoss())
    else:
        # integrated gradients RRRLoss
        reviseLoss = IGRRRLoss(net=net, regularizer_rate=20, base_criterion=BCELoss())

    # launch the debug a given number of iterations
    debug(
        net=net,
        dataloaders=dataloaders,
        cost_function=torch.nn.BCELoss(),
        optimizer=optimizer,
        title="debug",
        debug_train_loader=train_loader,
        debug_test_loader=val_loader,
        test_loader=test_loader,
        reviseLoss=reviseLoss,
        **vars(args)
    )

    print("After debugging...")

    # re-test set
    test_loss, test_accuracy, test_score = test_step(
        net=net,
        test_loader=iter(test_loader),
        cost_function=cost_function,
        title="Test",
        test=dataloaders["test"],
        device=args.device,
    )

    print(
        "\n\t [TEST SET]: Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
            test_loss, test_accuracy, test_score
        )
    )

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.log(
            {
                "test/post_loss": test_loss,
                "test/post_accuracy": test_accuracy,
                "test/post_score": test_score,
            }
        )

    # save the model state of the debugged network
    torch.save(
        net.state_dict(),
        os.path.join(
            args.model_folder,
            "debug_{}_{}.pth".format(
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
