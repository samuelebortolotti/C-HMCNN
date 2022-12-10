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
    """Loads the data from a pre-existing DataLoader
    (it should return the position of the confounder as well as the labels)
    In particular, this class aims to wrap the cifar dataloader with labels and confunder position
    and returns, a part from the label, the confounder mask which is useful for the RRR loss"""

    def __init__(
        self,
        train_set: LoadDataset,
    ):
        """Init param: it saves the training set"""
        self.train_set = train_set

    def __len__(self) -> int:
        """Returns the total amount of data.
        Returns:
            number of dataset entries [int]
        """
        return len(self.train_set)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, str, str]:
        """Returns the data, specifically:
        - element: train sample
        - hierarchical_label: hierarchical label
        - confounder_mask: mask for the RRR loss
        - counfounded: whether the sample is confounded or not
        - superclass: superclass in string
        - subclass: subclass in string

        Note that: the confounder mask is all zero for non-confounded examples
        """
        # get the data from the training_set
        (
            train_sample,
            superclass,
            subclass,
            hierarchical_label,
            confunder_pos1_x,
            confunder_pos1_y,
            confunder_pos2_x,
            confunder_pos2_y,
            confunder_shape,
        ) = self.train_set[idx]

        # parepare the train example and the explainations in the right shape
        single_el = prepare_single_test_sample(single_el=train_sample)

        # compute the confounder mask
        confounder_mask, confounded = compute_mask(
            shape=(train_sample.shape[1], train_sample.shape[2]),
            confunder_pos1_x=confunder_pos1_x,
            confunder_pos1_y=confunder_pos1_y,
            confunder_pos2_x=confunder_pos2_x,
            confunder_pos2_y=confunder_pos2_y,
            confunder_shape=confunder_shape,
        )

        # restore the requires grad flag
        single_el.requires_grad = False

        # returns the data
        return (
            single_el,
            hierarchical_label,
            confounder_mask,
            confounded,
            superclass,
            subclass,
        )


def save_sample(
    train_sample: torch.Tensor,
    prediction: torch.Tensor,
    idx: int,
    debug_folder: str,
    dataloaders: Dict[str, Any],
    superclass: str,
    subclass: str,
    integrated_gradients: bool,
) -> bool:
    """Save the test sample.
    Then it returns whether the sample contains a confunder or if the sample has been
    guessed correctly.

    Args:
        train_sample [torch.Tensor]: train sample depicting the image
        prediction [torch.Tensor]: prediction of the network on top of the test function
        idx [int]: index of the element of the batch to consider
        debug_folder [str]: string depicting the folder where to save the figures
        dataloaders [Dict[str, Any]]: dictionary depicting the dataloaders of the data
        superclass [str]: superclass string
        subclass [str]: subclass string
        integrated_gradients [bool]: whether to use integrated gradients

    Returns:
        correct_guess [bool]: whether the model has guessed correctly
    """
    # prepare the element in order to show it
    single_el_show = train_sample.squeeze(0).clone().cpu().data.numpy()
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

    # plot the title
    prediction_text = "Predicted: {}\nbecause of: {}".format(
        parent_predictions, children_predictions
    )

    # get figure
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
        "{}/iter_{}_original_{}{}.png".format(
            debug_folder,
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


def visualize_sample(
    single_el: torch.Tensor,
    debug_folder: str,
    idx: int,
    integrated_gradients: bool,
    confounder_mask: torch.Tensor,
    superclass: str,
    subclass: str,
    dataloaders: Dict[str, Any],
    device: str,
    net: nn.Module,
) -> None:
    """Save the samples information, including the sample itself, the gradients and the masked gradients

    Args:
        single_el [torch.Tensor]: sample to show
        debug_folder [str]: folder where to store the data
        idx [int]: idex of the element to save
        integrated_gradients [bool]: whether to use integrated gradiends
        confounder_mask [torch.Tensor]: confounder mask
        superclass [str]: superclass of the sample
        subclass [str]: subclass of the sample
        dataloaders [Dict[str, Any]]: dataloaders
        device [str]: device
        net [nn.Module]: network
    """
    # set the network to eval mode
    net.eval()

    # unsqueeze the element
    single_el = torch.unsqueeze(single_el, 0)

    # set the gradients as required
    single_el.requires_grad = True

    # move the element to device
    single_el = single_el.to(device)

    # get the predictions
    preds = net(single_el.float())

    # save the sample and whether the sample has been correctly guessed
    correct_guess = save_sample(
        train_sample=single_el,
        prediction=preds,
        idx=idx,
        debug_folder=debug_folder,
        dataloaders=dataloaders,
        superclass=superclass,
        subclass=subclass,
        integrated_gradients=integrated_gradients,
    )

    # compute the gradient, whether input or integrated
    gradient = compute_gradients(
        single_el=single_el,
        net=net,
        integrated_gradients=integrated_gradients,
    )

    # show the gradient
    show_gradient(
        gradient=gradient,
        confounder_mask=confounder_mask,
        debug_folder=debug_folder,
        idx=idx,
        correct_guess=correct_guess,
        integrated_gradients=integrated_gradients,
    )

    # show the masked gradient
    show_masked_gradient(
        confounder_mask=confounder_mask,
        gradient=gradient,
        debug_folder=debug_folder,
        idx=idx,
        integrated_gradients=integrated_gradients,
        correct_guess=correct_guess,
    )


def show_masked_gradient(
    confounder_mask: torch.Tensor,
    gradient: torch.Tensor,
    debug_folder: str,
    idx: int,
    integrated_gradients: bool,
    correct_guess: bool
) -> None:
    """Save the maked gradient

    Args:
        single_el [torch.Tensor]: sample to show
        debug_folder [str]: folder where to store the data
        idx [int]: idex of the element to save
        integrated_gradients [bool]: whether to use integrated gradiends
        confounder_mask [torch.Tensor]: confounder mask
        superclass [str]: superclass of the sample
        subclass [str]: subclass of the sample
        dataloaders [Dict[str, Any]]: dataloaders
        device [str]: device
        net [nn.Module]: network
    """
    # gradient to show
    gradient_to_show = gradient.clone().cpu().data.numpy()
    gradient_to_show = np.where(confounder_mask < 0.5, gradient_to_show, 0)
    gradient_to_show = np.fabs(gradient_to_show)
    # normalize the value
    gradient_to_show = gradient_to_show / np.max(gradient_to_show)

    # show the picture
    fig = plt.figure()
    plt.imshow(gradient_to_show, cmap="gray")
    plt.title("Gradient user modified: no confunder")
    # show the figure
    fig.savefig(
        "{}/iter_{}_gradient_no_confunder_{}{}.png".format(
            debug_folder,
            idx,
            "integrated" if integrated_gradients else "input",
            "_correct" if correct_guess else "",
        ),
        dpi=fig.dpi,
    )
    plt.close()


def show_gradient(
    gradient: torch.Tensor,
    confounder_mask: torch.Tensor,
    debug_folder: str,
    idx: int,
    correct_guess: bool,
    integrated_gradients: bool,
) -> None:
    """Save the gradient

    Args:
        gradient [torch.Tensor]: gradient to show
        confounder_mask [torch.Tensor]: confounder mask
        debug_folder [str]: folder where to store the data
        idx [int]: idex of the element to save
        correct_guess [bool]: whether the network got the example right
        integrated_gradients [bool]: whether to use integrated gradiends
    """
    # get the gradient
    gradient_to_show = gradient.clone().cpu().data.numpy()
    gradient_to_show = np.where(confounder_mask < 0.5, gradient_to_show, 0)
    gradient_to_show = np.fabs(gradient_to_show)

    # normalize the value
    gradient_to_show = gradient_to_show / np.max(gradient_to_show)

    # show the picture
    fig = plt.figure()
    plt.imshow(gradient_to_show, cmap="gray")
    plt.title("Gradient user modified: no confunder")
    # show the figure
    fig.savefig(
        "{}/iter_{}_gradient_no_confunder_{}{}.png".format(
            debug_folder,
            idx,
            "integrated" if integrated_gradients else "input",
            "_correct" if correct_guess else "",
        ),
        dpi=fig.dpi,
    )
    plt.close()


def prepare_single_test_sample(single_el: torch.Tensor) -> torch.Tensor:
    """Prepare the test sample
    It sets the gradient as required i order to make it compliant with the rest of the algorithm
    Args:
        single_el [torch.Tensor]: sample
    Returns:
        element [torch.Tensor]: single element
    """
    # requires grad
    single_el.requires_grad = True
    # returns element and predictions
    return single_el


def compute_gradients(
    single_el: torch.Tensor,
    net: nn.Module,
    integrated_gradients: bool,
) -> torch.Tensor:
    """Computes the integrated graditents or input gradient according to what the user specify in the argument

    Args:
        single_el [torch.Tensor]: sample depicting the image
        net [nn.Module]: neural network
        integrated_gradients [bool]: whether the method uses integrated gradients or input gradients
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

    return gradient


def compute_mask(
    shape: Tuple[int, int],
    confunder_pos1_x: int,
    confunder_pos1_y: int,
    confunder_pos2_x: int,
    confunder_pos2_y: int,
    confunder_shape: str,
) -> Tuple[torch.Tensor, bool]:
    """Compute the mask according to the confounder position and confounder shape

    Args:
        shape [Tuple[int, int]]: shape of the confounder mask
        confuder_pos1_x [int]: x of the starting point
        confuder_pos1_y [int]: y of the starting point
        confuder_pos2_x [int]: x of the ending point
        confuder_pos2_y [int]: y of the ending point
        confunder_shape [Dict[str, Any]]: confunder information

    Returns:
        confounder_mask [torch.Tensor]: tensor highlighting the area where the confounder is present with ones. It is zero elsewhere
        confounded [bool]: whether the sample is confounded or not
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

    # return the confounder mask and whether it s confounded
    return confounder_mask, confounded

def save_some_confounded_samples(
    net: nn.Module,
    start_from: int,
    number: int,
    loader: torch.utils.data.DataLoader,
    device: str,
    dataloaders: Dict[str, Any],
    folder: str,
    integrated_gradients: bool
) -> None:
    """Save some confounded examples according to the dataloader and the number of examples the user specifies

    Args:
        net [nn.Module]: neural network
        start_from [int]: start enumerating the sample from
        number [int]: stop enumerating the samples when the counter has reached (note that it has to start from start_from)
        loader [torch.utils.data.DataLoader]: loader where to save the samples
        device [str]: device
        dataloaders [Dict[str, Any]: dataloaders
        folder [str]: folder where to store the sample
        integrated_gradients [bool]: whether the gradient are integrated or input gradients
    """
    # set the networ to evaluation mode
    net.eval()

    # set the counter
    counter = start_from
    # set the done flag
    done = False
    for _, inputs in tqdm.tqdm(
        enumerate(loader),
        desc="Save",
    ):
        # get items
        (sample, _, confounder_mask, confounded, superclass, subclass) = inputs
        # loop over the batch
        for i in range(sample.shape[0]):
            # is the element confound?
            if confounded[i]:
                # visualize it
                visualize_sample(
                    sample[i],
                    folder,
                    counter,
                    integrated_gradients,
                    confounder_mask[i],
                    superclass[i],
                    subclass[i],
                    dataloaders,
                    device,
                    net,
                )
                # increase the counter
                counter += 1
                # stop when done
                if counter == number:
                    done = True
                    break
        # stop when done
        if done:
            break


def debug(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    debug_folder: str,
    iterations: int,
    cost_function: torch.nn.BCELoss,
    device: str,
    set_wandb: bool,
    integrated_gradients: bool,
    optimizer: torch.optim.Optimizer,
    debug_test_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    batch_size: int,
    test_batch_size: int,
    batches_treshold: float,
    reviseLoss: Union[RRRLoss, IGRRRLoss],
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
        debug_test_loader [torch.utils.data.DataLoader]: dataloader for the test
        batch_size [int]: size of the batch
        test_batch_size [int]: size of the batch in the test settings
        batches_treshold [float]: batches threshold
        reviseLoss: Union[RRRLoss, IGRRRLoss]: loss for feeding the network some feedbacks
        **kwargs [Any]: kwargs
    """
    print("Have to run for {} debug iterations...".format(iterations))

    # Load debug dataets
    debug_train = LoadDebugDataset(
        dataloaders["train_dataset_with_labels_and_confunders_position"],
    )
    debug_val = LoadDebugDataset(
        dataloaders["val_dataset_with_labels_and_confunders_position"]
    )
    test_debug = LoadDebugDataset(
        dataloaders["test_dataset_with_labels_and_confunders_pos"],
    )

    # Dataloaders
    debug_loader = torch.utils.data.DataLoader(
        debug_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    debug_val_loader = torch.utils.data.DataLoader(
        debug_val, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_debug = torch.utils.data.DataLoader(
        test_debug, batch_size=test_batch_size, shuffle=False, num_workers=4
    )

    # save some training samples (10 here)
    save_some_confounded_samples(
        net=net,
        start_from=0,
        number=10,
        dataloaders=dataloaders,
        device=device,
        folder=debug_folder,
        integrated_gradients=integrated_gradients,
        loader=iter(debug_loader),
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
            right_reason_loss_confounded,
        ) = revise_step(
            net=net,
            debug_loader=iter(debug_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=reviseLoss,
            device=device,
            title="Train with RRR",
            batches_treshold=batches_treshold
        )

        print(
            "\n\t Debug full loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, Score {:.5f}, Right Reason Loss on Confounded {:.5f}".format(
                total_loss,
                total_right_answer_loss,
                total_right_reason_loss,
                total_accuracy,
                total_score,
                right_reason_loss_confounded,
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "train/loss": total_loss,
                    "train/right_anwer_loss": total_right_answer_loss,
                    "train/right_reason_loss": total_right_reason_loss,
                    "train/accuracy": total_accuracy,
                    "train/score": total_score,
                    "train/confounded_samples_only_right_reason": right_reason_loss_confounded,
                }
            )

        # Validate with RRR feedbacks which are not used for training, but only for testing
        (
            val_loss,
            val_right_answer_loss,
            val_right_reason_loss,
            val_accuracy,
            val_score,
            val_right_reason_loss_confounded,
        ) = revise_step(
            net=net,
            debug_loader=iter(debug_val_loader),
            R=dataloaders["train_R"],
            train=dataloaders["train"],
            optimizer=optimizer,
            revive_function=reviseLoss,
            device=device,
            title="Debug with RRR",
            have_to_train = False,
            batches_treshold=batches_treshold
        )

        print(
            "\n\t Debug validation loss {:.5f}, Right Answer Loss {:.5f}, Right Reason Loss {:.5f}, Accuracy {:.2f}%, Score {:.5f}, Right Reason Loss on Confounded {:.5f}".format(
                val_loss,
                val_right_answer_loss,
                val_right_reason_loss,
                val_accuracy,
                val_score,
                val_right_reason_loss_confounded,
            )
        )

        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "val/loss": total_loss,
                    "val/right_anwer_loss": total_right_answer_loss,
                    "val/right_reason_loss": total_right_reason_loss,
                    "val/accuracy": total_accuracy,
                    "val/score": total_score,
                    "val/confounded_samples_only_right_reason": right_reason_loss_confounded,
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

        # test set
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

    # save some test confounded examples
    save_some_confounded_samples(
        net=net,
        start_from=100,
        number=110,
        dataloaders=dataloaders,
        device=device,
        folder=debug_folder,
        integrated_gradients=integrated_gradients,
        loader=iter(test_debug),
    )


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
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batches-treshold", type=float, default=float('inf'), help="batches treshold (infinity if not set)")
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
                "test/loss": test_loss,
                "test/accuracy": test_accuracy,
                "test/score": test_score,
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
                "test/loss": test_loss,
                "test/accuracy": test_accuracy,
                "test/score": test_score,
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
