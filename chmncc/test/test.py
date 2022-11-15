import torch
import torch.nn as nn
from typing import Tuple
import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def tr_image(img: torch.Tensor) -> torch.Tensor:
    r"""
    Function which computes the average of the image, to better visualize it
    in Tensor board

    Args:

    - img [torch.Tensor]: image

    Returns:
    - image after the processing [torch.Tensor]
    """
    return (img + 1) / 2


def test_step(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cost_function,
    writer: torch.utils.tensorboard.SummaryWriter,
    title: str,
    test,
    device="cuda",
) -> Tuple[float, float, float]:
    r""" """
    total = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network to evaluation mode
    net.eval()

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # iterate over the test set
        for batch_idx, (inputs, targets) in tqdm.tqdm(
            enumerate(test_loader), desc=title
        ):
            # load data into device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # forward pass
            outputs = net(inputs.float())
            # moving elements to cpu
            outputs = outputs.to("cpu")
            targets = targets.to("cpu")
            inputs = inputs.to("cpu")
            # predicted
            predicted = outputs.data > 0.5
            # total
            total += targets.shape[0] * targets.shape[1]
            # total correct predictions
            cumulative_accuracy += (predicted == targets.byte()).sum().item()

            # predicted
            predicted = outputs.data > 0.5

            # loss computation
            loss = cost_function(outputs, targets)

            # fetch prediction and loss value
            cumulative_loss += (
                loss.item()
            )  # Note: the .item() is needed to extract scalars from tensors

            # compute the au(prc)
            predicted = predicted.to("cpu")
            cpu_constrained_output = outputs.to("cpu")
            targets = targets.to("cpu")

            if batch_idx == 0:
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = targets
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, targets), dim=0)

    # average precision score
    score = average_precision_score(
        y_test[:, test.to_eval], constr_test.data[:, test.to_eval], average="micro"
    )

    return cumulative_loss / len(test_loader), cumulative_accuracy / total, score
