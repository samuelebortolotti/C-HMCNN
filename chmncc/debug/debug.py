from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import torch
import os
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from chmncc.networks import ResNet18, LeNet5
from chmncc.config import confunders, hierarchy
from chmncc.utils.utils import dotdict, load_best_weights
from torch.utils.data.sampler import SubsetRandomSampler
from chmncc.dataset import (
    dataloaders,
    load_cifar_dataloaders,
    get_named_label_predictions,
)
import wandb
from chmncc.test import test_step
from chmncc.explanations import compute_integrated_gradient, output_gradients
from chmncc.loss import RRRLoss, IGRRRLoss
from chmncc.revise import revise_step
from chmncc.optimizers import get_adam_optimizer
from chmncc.utils import dotdict
from typing import Dict, Any, Tuple
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchsummary import summary


def prepare_test_sample(
    net: nn.Module, sample_batch: torch.Tensor, idx: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract one element from the batch and sets gradients together with an additive element in the
    size in order to make it compliant with the rest of the algorithm.
    Args:
        net [nn.Module]: neural networks
        sample_batch [torch.Tensor]: batch of Test samples
        idx [int]: index of the element of the batch to consider
        device [str]: device
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
    # move the element to device
    single_el = single_el.to(device)
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
) -> bool:
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
    confunded = False
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
    return correct_guess


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
    single_el: torch.Tensor, net: nn.Module, debug_folder: str, idx: int, device: str
) -> torch.Tensor:
    """Computes the input graditents and saves them in an image

    Args:
        single_el [torch.Tensor]: test sample depicting the image
        net [nn.Module]: neural network
        debug_folder [str]: string depicting the folder path
        idx [int]: index of the image
        device [str]: device where to perform the computations
    Returns:
        gradient [torch.Tensor]: integrated_gradient
    """
    # move the element to cpu
    single_el = single_el.to("cpu")
    # integrated gradients
    gradient = output_gradients(single_el, net(single_el))[0]
    # move the element to cpu
    single_el = single_el.to(device)

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
    debug_folder: str,
    idx: int,
    gradient: torch.Tensor,
    confunder_pos1_x: int,
    confunder_pos1_y: int,
    confunder_pos2_x: int,
    confunder_pos2_y: int,
    confunder_shape: str,
) -> torch.Tensor:
    """Debug iteration
        - remove the confunder from the integrated gradient
        - saves the figure (integrated gradient without confunder)
        - revise the network (gives the feedback to the network)

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
    confounder_mask = np.zeros_like(gradient.cpu().data.numpy())

    if confunder_shape == "rectangle":
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
            (int(confunder_pos1_x.item()), int(confunder_pos1_y.item())),
            int(confunder_pos2_x.item()),
            (255, 255, 255),
            cv2.FILLED,
        )

    # binarize the mask and adjust it the right way
    confounder_mask = torch.tensor((confounder_mask > 0.5).astype(np.float_))

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
        "{}/{}_i_gradient_no_confunder.png".format(debug_folder, idx),
        dpi=fig.dpi,
    )
    plt.close()

    # return the confounder mask
    return confounder_mask


def debug(
    net: nn.Module,
    dataloaders: Dict[str, Any],
    debug_folder: str,
    device: str,
    set_wandb: bool,
    integrated_gradients: bool,
    optimizer: torch.optim.Optimizer,
    title: str,
    batch_size: int,
    test_batch_size: int,
    **kwargs: Any
):
    """Method which performs the debug step, by detecting the confunded images first;
    then correcting the integrated gradient associated and finally re-training the model with the RRR loss

    Args:
        net [nn.Module]: neural network
        dataloaders [Dict[str, Any]]: dataloaders
        debug_folder [str]: debug_folder
        device [str]: device
        integrated gradients [bool]: whether to use integrated gradients or not
        **kwargs [Any]: kwargs
    """

    print("#> Debug...")

    # log on wandb if and only if the module is loaded
    if set_wandb:
        wandb.watch(net)

    # load the human readable labels dataloader and the confunders position for debug and test (contains only confounders)
    test_set_confunder = dataloaders["test_dataset_with_labels_and_confunders_pos_only_confounders"]

    debug_train_loader = torch.utils.data.DataLoader(
        test_set_confunder, batch_size=batch_size
    )
    debug_test_loader = torch.utils.data.DataLoader(
        test_set_confunder, batch_size=test_batch_size
    )

    if not integrated_gradients:
        reviseLoss = RRRLoss(net=net, regularizer_rate=20, base_criterion=BCELoss())
    else:
        # integrated gradients RRRLoss
        reviseLoss = IGRRRLoss(net=net, regularizer_rate=20, base_criterion=BCELoss())

    # empty elements
    samples_list = None
    ground_truth_list = None
    confounder_mask_list = None

    print(len(iter(debug_train_loader)))
    # loop over the examples
    for _, inputs in tqdm.tqdm(enumerate(iter(debug_train_loader)), desc=title):
        print(_)
        (
            test_el,
            superclass,
            subclass,
            hierarchical_label,
            confunder_pos1_x,
            confunder_pos1_y,
            confunder_pos2_x,
            confunder_pos2_y,
            confunder_shape,
        ) = inputs

        # set the network to eval mode
        net.eval()
        continue

        # loop over the batch
        for i in range(test_el.shape[0]):
            # parepare the test example and the explainations in the right shape
            single_el, preds = prepare_test_sample(
                net=net, sample_batch=test_el, idx=i, device=device
            )
            # save the test sample only if it is the right one
            correct_guess = save_test_sample(
                test_sample=single_el,  # single torch image element
                prediction=preds,  # torch prediction
                idx=i,  # iteration
                debug_folder=debug_folder,  # string fdebug folde rname
                dataloaders=dataloaders,  # dictionary of dataloaders
                superclass=superclass[i],  # ith superclass string
                subclass=subclass[i],  # ith subclass string
            )

            # machine understands, keep looping
            if correct_guess:
                print("The machine start to understand something...")

            # get the example label
            label = torch.unsqueeze(hierarchical_label[i], 0)

            # save the integrated gradients
            if integrated_gradients:
                gradient = save_i_gradient(
                    single_el=single_el,
                    net=net,
                    debug_folder=debug_folder,
                    idx=i,
                )
            else:
                gradient = save_input_gradient(
                    single_el=single_el,
                    net=net,
                    debug_folder=debug_folder,
                    idx=i,
                    device=device,
                )

            # debug iteration
            confounder_mask = debug_iter(
                debug_folder=debug_folder,
                idx=i,
                gradient=gradient,
                confunder_pos1_x=confunder_pos1_x[i],
                confunder_pos1_y=confunder_pos1_y[i],
                confunder_pos2_x=confunder_pos2_x[i],
                confunder_pos2_y=confunder_pos2_y[i],
                confunder_shape=confunder_shape[i],
            )
            confounder_mask = torch.unsqueeze(confounder_mask, 0)

            if (
                samples_list == None
                or ground_truth_list == None
                or confounder_mask_list == None
            ):
                samples_list = single_el
                ground_truth_list = label
                confounder_mask_list = confounder_mask
            else:
                samples_list = torch.cat((samples_list, single_el), 0)
                ground_truth_list = torch.cat((ground_truth_list, label), 0)
                confounder_mask_list = torch.cat(
                    (confounder_mask_list, confounder_mask), 0
                )

        # check whether the model can be trained in this way
        if (
            samples_list == None
            or len(samples_list) == 1
            or ground_truth_list == None
            or len(ground_truth_list) == 1
            or confounder_mask_list == None
            or len(confounder_mask_list) == 1
        ):
            print("Not enough examples to train the model")
            continue

        loss, right_answer_loss, right_reason_loss, accuracy = revise_step(
            net=net,
            training_samples=samples_list,
            ground_truths=ground_truth_list,
            confunder_masks=confounder_mask_list,
            R=dataloaders["train_R"],
            train=dataloaders["test"],
            optimizer=optimizer,
            revive_function=reviseLoss,
            device=device,
        )
        print(
            "Loss",
            loss,
            "RAL",
            right_answer_loss,
            "RRL",
            right_reason_loss,
            "ACC",
            accuracy,
        )
        # log on wandb if and only if the module is loaded
        if set_wandb:
            wandb.log(
                {
                    "debug/rrr_loss": loss,
                    "debug/right_anwer_loss": right_answer_loss,
                    "debug/right_reason_loss": right_reason_loss,
                    "debug/accuracy": accuracy,
                }
            )

        # reset the lists
        samples_list = None
        ground_truth_list = None
        confounder_mask_list = None

    print("Done with debug")

    print("-----------------------------------------------------")

    print("Testing...")

    # define the cost function
    cost_function = torch.nn.BCELoss()

    # test set
    test_loss, test_accuracy, test_score = test_step(
        net=net,
        test_loader=iter(debug_test_loader),
        cost_function=cost_function,
        title="Test",
        test=dataloaders["test"],
        device=device,
        debug_mode=True
    )

    print("Netowrk resumed, performances:")

    print(
        "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
            test_loss, test_accuracy, test_score
        )
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
        action="store_true",
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
        test_debug_csv_path="./dataset/test_debug.csv",
        test_debug_test_csv_path="./dataset/test_debug_test.csv",
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
    test_loader = dataloaders["test_loader"]

    # define the cost function
    cost_function = torch.nn.BCELoss()

    #  # test set
    #  test_loss, test_accuracy, test_score = test_step(
    #      net=net,
    #      test_loader=iter(test_loader),
    #      cost_function=cost_function,
    #      title="Test",
    #      test=dataloaders["test"],
    #      device=args.device,
    #  )
    #
    #  print("Network resumed, performances:")
    #
    #  print(
    #      "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
    #          test_loss, test_accuracy, test_score
    #      )
    #  )

    print("-----------------------------------------------------")

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.log(
            {
                "debug_test/loss": test_loss,
                "debug_test/accuracy": test_accuracy,
                "debug_test/score": test_score,
            }
        )

    # launch the debug
    debug(
        net=net,
        dataloaders=dataloaders,
        optimizer=optimizer,
        title="debug",
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
        "\n\t Test loss {:.5f}, Test accuracy {:.2f}%, Test Area under Precision-Recall Curve {:.3f}".format(
            test_loss, test_accuracy, test_score
        )
    )

    # log on wandb if and only if the module is loaded
    if args.wandb:
        wandb.log(
            {
                "debug_test/loss": test_loss,
                "debug_test/accuracy": test_accuracy,
                "debug_test/score": test_score,
            }
        )

    # close wandb
    if args.wandb:
        # finish the log
        wandb.finish()
