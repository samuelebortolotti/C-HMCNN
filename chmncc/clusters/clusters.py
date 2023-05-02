"""Train models on several clusters"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from chmncc.dataset.load_dataset import LoadDataset
import torch
import os
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
from chmncc.networks import ResNet18, LeNet5, LeNet7, AlexNet, MLP
from chmncc.optimizers.adam import get_adam_optimizer_with_gate
from chmncc.utils.utils import (
    load_best_weights,
    load_best_weights_gate,
    get_lr,
    prepare_probabilistic_circuit,
)
from chmncc.dataset import (
    load_dataloaders,
)
from chmncc.optimizers import get_adam_optimizer, get_plateau_scheduler
from typing import Dict, Any, List
import tqdm
import random
import copy
from torchsummary import summary
from chmncc.train import training_step
from chmncc.train.train import training_step_with_gate

from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running train on subclusters of the network
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("clusters", help="Train models on several clusters")
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        choices=["resnet", "lenet", "lenet7", "alexnet", "mlp"],
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
        "--number-of-splits",
        "-nos",
        type=int,
        default=30,
        help="Number of splits for clusters",
    )
    parser.add_argument(
        "--number-of-epochs",
        "-noe",
        type=int,
        default=5,
        help="Number of epoch for each cluster",
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="Train batch size"
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
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device",
    )
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
        "--num-workers", type=int, default=4, help="dataloaders num workers"
    )
    parser.add_argument(
        "--prediction-treshold",
        type=float,
        default=0.5,
        help="considers the class to be predicted in a multilabel classification setting",
    )
    parser.add_argument("--patience", type=int, default=5, help="scheduler patience")
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar",
        choices=["mnist", "cifar", "fashion", "omniglot"],
        help="which dataset to use",
    )
    parser.add_argument(
        "--simplified-dataset",
        "-simdat",
        dest="simplified_dataset",
        action="store_true",
        help="If possibile, use a simplified version of the dataset",
    )
    parser.add_argument(
        "--imbalance-dataset",
        "-imdat",
        dest="imbalance_dataset",
        action="store_true",
        help="Imbalance the dataset introducing another type of confounding",
    )
    parser.add_argument(
        "--use-probabilistic-circuits",
        "-probcirc",
        action="store_true",
        help="Whether to use the probabilistic circuit instead of the Giunchiglia approach",
    )
    parser.add_argument(
        "--gates",
        type=int,
        default=1,
        help="Number of hidden layers in gating function (default: 1)",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=1,
        help="Number of hidden layers in gating function (default: 1)",
    )
    parser.add_argument(
        "--S", type=int, default=0, help="PSDD scaling factor (default: 0)"
    )
    parser.add_argument(
        "--constraint-folder",
        type=str,
        default="./constraints",
        help="Folder for storing the constraints",
    )
    parser.add_argument("--fine-tune", action="store_true", help="Fine tune the model")
    # set the main function to run when blob is called from the command line
    parser.set_defaults(
        func=main,
        constrained_layer=True,
        force_prediction=False,
        fixed_confounder=False,
        use_softmax=False,
        simplified_dataset=False,
        imbalance_dataset=False,
        fine_tune=False,
    )


def get_network(
    network: str,
    dataloaders: Dict[str, Any],
    output_classes: int,
    constrained_layer: bool,
    img_depth: int,
    img_size: int,
) -> nn.Module:
    # Network
    if network == "lenet":
        net = LeNet5(
            dataloaders["train_R"], output_classes, constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    elif network == "lenet7":
        net = LeNet7(
            dataloaders["train_R"], output_classes, constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    elif network == "alexnet":
        net = AlexNet(
            dataloaders["train_R"], output_classes, constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    elif network == "mlp":
        net = MLP(
            dataloaders["train_R"],
            output_classes,
            constrained_layer,
            channels=img_depth,
            img_height=img_size,
            img_width=img_size,
        )  # 20 superclasses, 100 subclasses + the root
    else:
        net = ResNet18(
            dataloaders["train_R"], 121, constrained_layer
        )  # 20 superclasses, 100 subclasses + the root
    return net


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def organize_clusters(dataset: LoadDataset, num_splits: int) -> List[LoadDataset]:
    data_list_dict = {}
    dataloader_list = list()

    # create the list of data divided by subclasses
    for item, superclass, subclass in dataset.data_list:
        if superclass not in data_list_dict:
            data_list_dict[superclass] = list()
        data_list_dict[superclass].append((item, superclass, subclass))

    for key in data_list_dict:
        print("Prima", len(data_list_dict[key]))
        random.shuffle(data_list_dict[key])
        data_list_dict[key] = list(split_list(data_list_dict[key], num_splits))
        print("Dopo", len(data_list_dict[key]))

    for i in range(num_splits):
        new_dataset = copy.deepcopy(dataset)
        new_dataset.data_list = list()
        for key in data_list_dict:
            new_dataset.data_list.extend(data_list_dict[key][i])

        dataloader_list.append(new_dataset)
        print("Finito", len(new_dataset.data_list))

    return dataloader_list


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

    # img_size
    img_size = 32
    img_depth = 3
    output_classes = 121

    if args.dataset == "mnist":
        img_size = 28
        img_depth = 1
        output_classes = 67
    elif args.dataset == "fashion":
        img_size = 28
        img_depth = 1
        output_classes = 10
    elif args.dataset == "omniglot":
        img_size = 32
        img_depth = 1
        output_classes = 680

    if args.network == "alexnet":
        img_size = 224

    # Load dataloaders
    dataloaders = load_dataloaders(
        dataset_type=args.dataset,
        img_size=img_size,  # the size is squared
        img_depth=img_depth,  # number of channels
        device=args.device,
        csv_path="./dataset/train.csv",
        test_csv_path="./dataset/test_reduced.csv",
        val_csv_path="./dataset/val.csv",
        cifar_metadata="./dataset/pickle_files/meta",
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        normalize=True,  # normalize the dataset
        num_workers=args.num_workers,
        fixed_confounder=args.fixed_confounder,
        simplified_dataset=args.simplified_dataset,
        imbalance_dataset=args.imbalance_dataset,
    )

    # divide the datasets
    print("Train dataset statistics:")
    dataloaders["train_set"].print_stats()
    dataset = dataloaders["train_set"]
    # get the clusters
    clusters = organize_clusters(dataset, args.number_of_splits)

    # loop for the number of epochs required
    for s in range(args.number_of_splits):
        # get the loader out of the clusters
        train_loader = torch.utils.data.DataLoader(
            clusters[s],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        # move everything on the cpu
        net = get_network(
            args.network,
            dataloaders,
            output_classes,
            args.constrained_layer,
            img_depth,
            img_size,
        )
        print("Network", args.network)
        net = net.to(args.device)
        net.R = net.R.to(args.device)
        # zero grad
        net.zero_grad()
        summary(net, (img_depth, img_size, img_size))

        # set wandb if needed
        # use the probabilistic circuit
        cmpe: CircuitMPE = None
        gate: DenseGatingFunction = None

        if args.use_probabilistic_circuits:
            print("Using probabilistic circuits...")
            cmpe, gate = prepare_probabilistic_circuit(
                dataloaders["train_set"].get_A(),
                args.constraint_folder,
                args.dataset,
                args.device,
                args.gates,
                args.num_reps,
                output_classes,
                args.S,
            )

        # optimizer
        optimizer = get_adam_optimizer(
            net, args.learning_rate, weight_decay=args.weight_decay
        )
        if args.use_probabilistic_circuits:
            optimizer = get_adam_optimizer_with_gate(
                net, gate, args.learning_rate, weight_decay=args.weight_decay
            )

        # scheduler
        scheduler = get_plateau_scheduler(optimizer=optimizer, patience=args.patience)

        # Test on best weights (of the confounded model)
        if args.fine_tune:
            load_best_weights(net, args.weights_path_folder, args.device)
            if args.use_probabilistic_circuits:
                load_best_weights_gate(gate, args.weights_path_folder, args.device)

        cost_function = BCELoss()

        # for each epoch, train the network and then compute evaluation results
        for e in tqdm.tqdm(range(args.number_of_epochs), desc="Epochs"):

            train_loss: float = 0.0
            train_score: float = 0.0
            train_accuracy: float = 0.0
            train_au_prc_score_const: float = 0.0
            train_jaccard: float = 0.0
            train_hamming: float = 0.0
            train_au_prc_score_raw: float = 0.0

            # training step
            if args.use_probabilistic_circuits:
                (
                    train_loss,
                    train_accuracy,
                    train_jaccard,
                    train_hamming,
                    train_score,
                ) = training_step_with_gate(
                    net=net,
                    gate=gate,
                    cmpe=cmpe,
                    train=dataloaders["train"],
                    train_loader=train_loader,
                    optimizer=optimizer,
                    title="Training",
                    device=args.device,
                )
            else:
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
                    device=args.device,
                    constrained_layer=args.constrained_layer,
                    prediction_treshold=args.prediction_treshold,
                    force_prediction=args.force_prediction,
                    use_softmax=args.use_softmax,
                    superclasses_number=dataloaders["train_set"].n_superclasses,
                )

            print("Learning rate:", get_lr(optimizer))

            # test value
            print("\nEpoch: {:d}".format(e + 1))

            if args.use_probabilistic_circuits:
                print(
                    "\t Training loss {:.5f}, Training accuracy {:.2f}%, Training Jaccard Score {:.3f}, Training Hamming Loss {:.3f}, Training Area under Precision-Recall Curve Raw {:.3f}".format(
                        train_loss,
                        train_accuracy,
                        train_jaccard,
                        train_hamming,
                        train_score,
                    )
                )
            else:
                print(
                    "\t Training loss {:.5f}, Training accuracy {:.2f}%, Training Area under Precision-Recall Curve Raw {:.3f}, Training Area under Precision-Recall Curve Const {:.3f}".format(
                        train_loss,
                        train_accuracy,
                        train_au_prc_score_raw,
                        train_au_prc_score_const,
                    )
                )
            print("-----------------------------------------------------")

            # update scheduler
            scheduler.step(train_loss)

        print("Saving model on split {}...".format(s))
        torch.save(
            net.state_dict(),
            os.path.join(args.model_folder, "split_{}_net.pth".format(s)),
        )
        if args.use_probabilistic_circuits:
            torch.save(
                gate.state_dict(),
                os.path.join(args.model_folder, "split_{}_gate.pth".format(s)),
            )
