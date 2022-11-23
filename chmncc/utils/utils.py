import torch
import torch.nn as nn
import os
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils import tensorboard

################### Dotdict ##################


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


################### Get Lr  ##################


def get_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    r"""
    Function which returns the learning rate value
    used in the optimizer

    Args:
        optimizer [torch.optim.Optimizer]: optimizer

    Returns:
        lr [float]: learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


################### Logging ###################


def log_images(
    writer: tensorboard.SummaryWriter,
    img: torch.Tensor,
    epoch: int,
    title: str,
) -> None:
    r"""
    Log images on the summary writer, this function is usefull during
    debug sessions.

    Args:
        writer [tensorboard.SummaryWriter]: summary writer
        img [torch.Tensor]: image to log
        title [str]: title of the log
    """
    try:
        # Log training images in a row of 8
        logged_img = torchvision.utils.make_grid(
            img.data, nrow=8, normalize=True, scale_each=True
        )
        # add image
        writer.add_image(title, logged_img, epoch)
    except Exception as e:
        print("Couldn't log results: {}".format(e))


def log_values(
    writer: tensorboard.SummaryWriter,
    step: int,
    loss: float,
    accuracy: float,
    prefix: str,
) -> None:
    r"""
    Function which writes the loss and the accuracy of a model by the means of
    a SummaryWriter

    Args:
        writer [tensorboard.SummaryWriter]: summary writer
        step [int]: current step
        loss [float]: current loss
        accuracy [float]: accuracy
        prefix [str]: log prefix
    """
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/accuracy", accuracy, step)


################### Resume the network ###################


def load_best_weights(net: nn.Module, exp_name: str, device: str) -> None:
    r"""
    Function which loads the best weights of the network, basically it
    looks for the `{exp_name}/best.pth` and loads it

    Args:
        net [nn.Module]: network architecture
        exp_name [str]: folder name
        device [str]: device name
    """
    best_file = os.path.join(exp_name, "best.pth")
    if os.path.isfile(best_file):
        checkpoint = torch.load(best_file, map_location=torch.device(device))
        print("#> Resume best")
        net.load_state_dict(checkpoint)
    else:
        print("## Not Resumed ##")


def resume_training(
    resume_training: bool,
    experiment_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[Dict, Dict, int]:
    r"""
    Resumes the training if the corresponding flag is specified.
    If the resume_training flag is set to true, the function tries to recover, from the
    checkpoint specified, the number of epoch, training and validation parameters.
    If the resume_training flag is set to false, the parameters are set to the default ones

    Args:
        model [nn.Module]: network architecture
        experiment_name [str]: where the weight file is located
        optimizer [torch.optim.Optimizer]: optimizer

    Returns:
        training_params [Dict]: training parameters
        val_params [Dict]: validation parameters [step, best_loss]
        start_epoch [int]: start epoch number
    """
    if resume_training:
        print(f"#> Recovering the last paramters at {experiment_name}/ckpt.pth")
        resumef = os.path.join(experiment_name, "ckpt.pth")
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("#> Resuming previous training")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            training_params = checkpoint["training_params"]
            start_epoch = training_params["start_epoch"] + 1
            val_params = checkpoint["val_params"]
            print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint["optimizer"]["param_groups"]))
            print("==> checkpoint['training_params']")
            for k in checkpoint["training_params"]:
                print("\t{}, {}".format(k, checkpoint["training_params"][k]))
            print("==> checkpoint['val_params']")
            for k in checkpoint["val_params"]:
                print("\t{}, {}".format(k, checkpoint["val_params"][k]))
        else:
            raise Exception(
                "Couldn't resume training with checkpoint {}".format(resumef)
            )
    else:
        start_epoch = 0
        training_params = {}
        val_params = {}
        # worst possible auc score
        val_params["best_score"] = -np.inf

    return training_params, val_params, start_epoch


########## GET CONSTRAINED OUTPUT ######################:w


def get_constr_out(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Given the output of the neural network x returns the output of MCM given the hierarchy constraint
    expressed in the matrix R. This is employed only during the evaluation/test phase of the network

    Args:
        x [torch.tensor]: output of the neural network
        R [torch.tensor]: matrix of the ancestors

    Returns:
        final_out [torch.tensor]: output constrained
    """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out


def average_image_contributions(image: np.ndarray) -> np.ndarray:
    """Returns a black and white image making the average contribution of each channel

    Args:
        image [np.ndarray]: numpy array of size [size1, ...., channels]
    Returns:
        average [np.ndarray] image
    """
    return np.mean(image, axis=image.ndim - 1)
