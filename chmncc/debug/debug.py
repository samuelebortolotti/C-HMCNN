from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from chmncc.networks import ResNet18, LeNet5
from chmncc.config import confunders, hierarchy
from chmncc.utils.utils import load_best_weights, average_image_contributions_tensor
from chmncc.dataset import (
    load_cifar_dataloaders,
    get_named_label_predictions,
)
from chmncc.explanations import compute_integrated_gradient, output_gradients
from chmncc.loss import RRRLoss, IGRRRLoss
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchsummary import summary


def prepare_test_sample(
    net: nn.Module, sample_batch: torch.Tensor, idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract one element from the batch and sets gradients together with an additive element in the
    size in order to make it compliant with the rest of the algorithm.
    Args:
        net [nn.Module]: neural networks
        sample_batch [torch.Tensor]: batch of Test samples
        idx [int]: index of the element of the batch to consider
    Returns:
        element [torch.Tensor]: single element
        preds [torch.Tensor]: prediction
    """
    # set the model in eval mode
    net.eval()
    # get the single element batch
    single_el = torch.unsqueeze(sample_batch[idx], 0)
    # set the gradients as required
    single_el.requires_grad = True
    # get the predictions
    preds = net(single_el.float())
    # returns element and predictions
    return single_el, preds


def save_test_sample(
    test_sample: torch.Tensor,
    prediction: torch.Tensor,
    idx: int,
    debug_folder: str,
    dataloaders: Dict[str, Any],
    superclass: str,
    subclass: str,
) -> Tuple[bool, bool]:
    """Save the test sample only if it presents a confunder.
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

    Returns:
        confunded [bool]: whether the sample is confunded
        correct_guess [bool]: whether the model has guessed correctly
    """
    # prepare the element in order to show it
    single_el_show = test_sample[0].clone().cpu().data.numpy()
    single_el_show = single_el_show.transpose(1, 2, 0)

    # normalize
    single_el_show = np.fabs(single_el_show)
    single_el_show = single_el_show / np.max(single_el_show)

    # get the prediction
    predicted_1_0 = prediction.data > 0.5
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
    confunded = False
    correct_guess = False

    # set the guess as correct if it is
    if len(parent_predictions) == 1 and len(children_predictions) == 1:
        if (
            superclass.strip() == parent_predictions[0].strip()
            and subclass == children_predictions[0].strip()
        ):
            correct_guess = True

    # check if the sample is confunded
    if superclass in confunders:
        for tmp_index in range(len(confunders[superclass]["test"])):
            print(superclass, idx)
            if confunders[superclass]["test"][tmp_index]["subclass"] == subclass:
                print("Found confunder!")
                confunded = True
                break

    # if confunded, it is worth to save it
    if confunded:
        # plot the title
        prediction_text = "Predicted: {}\nbecause of: {}".format(
            parent_predictions, children_predictions
        )
        print("Confunders in the image found, saving the image....")
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
            "{}/{}_original{}.png".format(
                debug_folder,
                idx,
                "_confunded" if confunded else "",
            ),
            dpi=fig.dpi,
        )
        # close the figure
        plt.close()

    # whether the image is confunded or not
    return confunded, correct_guess


def save_i_gradient(
    single_el: torch.Tensor, net: nn.Module, debug_folder: str, idx: int
) -> torch.Tensor:
    """Computes the integrated graditents and saves them in an image

    Args:
        single_el [torch.Tensor]: test sample depicting the image
        net [nn.Module]: neural network
        debug_folder [str]: string depicting the folder path
        idx [int]: index of the image
    Returns:
        integrated_gradient [torch.Tensor]: integrated_gradient
    """
    # integrated gradients
    i_gradient = compute_integrated_gradient(
        single_el, torch.zeros_like(single_el), net
    )
    # sum over RGB channels
    i_gradient = torch.sum(i_gradient, dim=0)

    # permute to show
    i_gradient_to_show = i_gradient.clone()
    # get the absolute value
    i_gradient_to_show = torch.abs(i_gradient_to_show)
    # normalize the value
    i_gradient_to_show = i_gradient_to_show / torch.max(i_gradient_to_show)

    # show
    fig = plt.figure()
    plt.imshow(i_gradient_to_show.cpu().data.numpy(), cmap="gray")
    plt.title("Integrated Gradient with respect to the input")
    # show the figure
    fig.savefig(
        "{}/{}_i_gradient.png".format(debug_folder, idx),
        dpi=fig.dpi,
    )
    plt.close()

    # return the integrated gradients
    return i_gradient


def save_input_gradient(
    single_el: torch.Tensor, net: nn.Module, debug_folder: str, idx: int
) -> torch.Tensor:
    """Computes the input graditents and saves them in an image

    Args:
        single_el [torch.Tensor]: test sample depicting the image
        net [nn.Module]: neural network
        debug_folder [str]: string depicting the folder path
        idx [int]: index of the image
    Returns:
        gradient [torch.Tensor]: integrated_gradient
    """
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

    # show
    fig = plt.figure()
    plt.imshow(gradient_to_show.cpu().data.numpy(), cmap="gray")
    plt.title("Input Gradient with respect to the input")
    # show the figure
    fig.savefig(
        "{}/{}_i_gradient.png".format(debug_folder, idx),
        dpi=fig.dpi,
    )
    plt.close()

    # return the gradients
    return gradient


def debug_iter(
    net: nn.Module,
    sample_test: torch.Tensor,
    ground_truth: torch.Tensor,
    preds: torch.Tensor,
    reviveLoss: RRRLoss,
    debug_folder: str,
    idx: int,
    gradient: torch.Tensor,
    confunder_pos1_x: int,
    confunder_pos1_y: int,
    confunder_pos2_x: int,
    confunder_pos2_y: int,
    confunder_shape: Dict[str, str],
) -> None:
    """Debug iteration
        - remove the confunder from the integrated gradient
        - saves the figure (integrated gradient without confunder)
        - revives the network (gives the feedback to the network)

    Args:
        net [nn.Module]: neural network
        sample_test [torch.Tensor]: sample element tensor
        ground_truth [torch.Tensor]: ground truth label of the tensor
        preds [torch.Tensor]: prediction of the network given the input
        debug_folder [str]: string depicting the folder path
        idx [int]: index of the image
        gradient [torch.Tensor]: integrated gradient tensor
        confuder_pos1_x [int]: x of the starting point
        confuder_pos1_y [int]: y of the starting point
        confuder_pos2_x [int]: x of the ending point
        confuder_pos2_y [int]: y of the ending point
        confunder_shape [Dict[str, Any]]: confunder information
    """
    # confounder mask
    confounder_mask = np.zeros_like(gradient.data.numpy())

    print(confounder_mask.shape, gradient.shape)

    if confunder_shape["shape"] == "rectangle":
        # get the image of the modified gradient
        cv2.rectangle(
            confounder_mask,
            (confunder_pos1_x.item(), confunder_pos1_y.item()),
            (confunder_pos2_x.item(), confunder_pos2_y.item()),
            (255, 255, 255),
            cv2.FILLED,
        )
    else:
        # get the image of the modified gradient
        cv2.circle(
            confounder_mask,
            (confunder_pos1_x.numpy(), confunder_pos1_y.numpy()),
            confunder_pos2_x.numpy(),
            (255, 255, 255),
            cv2.FILLED,
        )

    # binarize the mask and adjust it the right way
    confounder_mask = torch.tensor((confounder_mask > 0.5).astype(np.float_))

    # gradient to show
    gradient_to_show = gradient.clone().data.numpy()
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
        "{}/{}_i_gradient_no_confunder.png".format(debug_folder, idx),
        dpi=fig.dpi,
    )
    plt.close()

    # compute the RRR loss
    print("Ground_truth", ground_truth.shape)
    print("Predictions", preds.shape)
    loss, ra_loss_c, rr_loss_c = reviveLoss(
        X=sample_test, y=ground_truth, expl=confounder_mask, logits=preds
    )
    print(loss, ra_loss_c, rr_loss_c)


def debug(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    debug_folder: str,
    integrated_gradients: bool,
    **kwargs: Any
):
    """Method which performs the debug step, by detecting the confunded images first;
    then correcting the integrated gradient associated and finally re-training the model with the RRR loss

    Args:
        net [nn.Module]: neural network
        dataloaders [Dict[str, Any]]: dataloaders
        debug_folder [str]: debug_folder
        integrated gradients [bool]: whether to use integrated gradients or not
        **kwargs [Any]: kwargs
    """

    print("#> Debug...")

    # load the human readable labels dataloader and the confunders position
    test_loader_with_label_names = dataloaders[
        "test_loader_with_labels_name_confunders_pos"
    ]

    # extract also the names of the classes
    (
        test_el,  # test image
        superclass,  # str superclass
        subclass,  # str subclass
        hierarchical_label,  # ground_truth
        confunder_pos1_x,  # integer first point second coordinate
        confunder_pos1_y,  # integer first point second coordinate
        confunder_pos2_x,  # integer second point first coordinate
        confunder_pos2_y,  # integer second point second coordinate
        confunder_shape,  # dictionary of the shape information
    ) = next(iter(test_loader_with_label_names))

    # if integrated gradients are not available then go for the classical gradients
    if not integrated_gradients:
        reviveLoss = RRRLoss(net=net, regularizer_rate=20, base_criterion=BCELoss())
    else:
        # integrated gradients RRRLoss
        reviveLoss = IGRRRLoss(net=net, regularizer_rate=20, base_criterion=BCELoss())

    # loop over the examples
    for i in range(test_el.shape[0]):
        # parepare the test example and the explainations in the right shape
        single_el, preds = prepare_test_sample(net=net, sample_batch=test_el, idx=i)

        # save the test sample only if it is the right one
        confunded_sample, correct_guess = save_test_sample(
            test_sample=single_el,  # single torch image element
            prediction=preds,  # torch prediction
            idx=i,  # iteration
            debug_folder=debug_folder,  # string fdebug folde rname
            dataloaders=dataloaders,  # dictionary of dataloaders
            superclass=superclass[i],  # ith superclass string
            subclass=subclass[i],  # ith subclass string
        )

        # confunded sample found, starting the debug procedure
        if confunded_sample:
            print("Confund!")
            # machine understands, keep looping
            if correct_guess:
                print("The machine start to understand something...")
                continue

            # get the example label
            label = torch.unsqueeze(hierarchical_label[i], 0)

            # save the integrated gradients
            if integrated_gradients:
                gradient = save_i_gradient(
                    single_el=single_el, net=net, debug_folder=debug_folder, idx=i
                )
            else:
                gradient = save_input_gradient(
                    single_el=single_el, net=net, debug_folder=debug_folder, idx=i
                )

            # debug iteration
            debug_iter(
                net=net,
                sample_test=single_el,  # single torch image element
                ground_truth=label,  # hierarchical label
                preds=preds,
                reviveLoss=reviveLoss,
                debug_folder=debug_folder,
                idx=i,
                gradient=gradient,
                confunder_pos1_x=confunder_pos1_x[i],
                confunder_pos1_y=confunder_pos1_y[i],
                confunder_pos2_x=confunder_pos2_x[i],
                confunder_pos2_y=confunder_pos2_y[i],
                confunder_shape=confunder_shape[i],
            )

    print("Done with debug")


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
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="Train batch size"
    )
    parser.add_argument(
        "--debug-folder", "-df", type=str, default="debug", help="Debug folder"
    )
    parser.add_argument(
        "--test-batch-size", "-tbs", type=int, default=128, help="Test batch size"
    )
    parser.add_argument(
        "--integrated-gradients",
        "-igrad",
        type=bool,
        default=False,
        help="Use integrated gradients",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Test batch size",
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


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
    net = net.to("cpu")
    net.R = net.R.to("cpu")
    # show a summary
    summary(net, (3, 32, 32))

    # Test on best weights
    load_best_weights(net, args.weights_path_folder, args.device)

    # dataloaders
    #  test_loader = dataloaders["test_loader"]

    # define the cost function
    #  cost_function = torch.nn.BCELoss()

    # test set
    #  test_loss, test_accuracy, test_score = test_step(
    #      net=net,
    #      test_loader=iter(test_loader),
    #      cost_function=cost_function,
    #      title="Test",
    #      test=dataloaders["test"],
    #      device=args.device,
    #  )
    #
    #  print("Netowrk resumed, performances:")
    #
    #  print(
    #      "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
    #          test_loss, test_accuracy, test_score
    #      )
    #  )

    print("-----------------------------------------------------")

    # zero grad
    net.zero_grad()

    # launch the debug
    debug(net, dataloaders, **vars(args))
