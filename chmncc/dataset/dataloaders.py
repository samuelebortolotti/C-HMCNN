from typing import Tuple, List, Dict, Any, Optional
import torchvision
import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from chmncc.utils import dotdict
from chmncc.dataset import (
    initialize_dataset,
    initialize_other_dataset,
    datasets,
)
from sklearn.model_selection import train_test_split
from chmncc.dataset.load_dataset_factory import LoadDatasetFactory
from chmncc.config import (
    cifar_hierarchy,
    mnist_hierarchy,
    fashion_hierarchy,
    omniglot_hierarchy,
)

#### Compute Mean and Stdev ################


def get_mean_std(
    img_size: Tuple[int, int], source_data: str
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    r"""
    Computes mean and standard deviation over the source and
    target dataloader

    This function has been adapted from
    [link]: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py

    Args:
        img_size [Tuple[int, int]]: image shape
        source_data [str]: path to the source data
    Returns:
        source mean and stdev Tuple[float, float]
    """

    # basic transformations
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ]
    )

    # datasets
    source_data = torchvision.datasets.ImageFolder(source_data, transform=transform)

    # data loader
    source_loader = torch.utils.data.DataLoader(
        dataset=source_data, batch_size=64, shuffle=True
    )

    def compute_mean_std(loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        r"""
        Computes mean and standard deviation over the source and
        target dataloader

        Args:
            loader [torch.utils.data.DataLoader]: dataloader

        Returns:
            dataloader mean and stdev [Tuple[float, float]]
        """
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

        for data, _ in tqdm(loader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

        return mean, std

    return compute_mean_std(source_loader)


############ Load dataloaders ############


def load_dataloaders(
    dataset_type: str,
    img_size: int,
    img_depth: int,
    csv_path: str,
    test_csv_path: str,
    val_csv_path: str,
    cifar_metadata: str,
    device: str,
    batch_size: int = 128,
    test_batch_size: int = 256,
    mean: List[float] = [
        0.5074,
        0.4867,
        0.4411,
    ],
    stdev: List[float] = [0.2011, 0.1987, 0.2025],
    additional_transformations: Optional[List[Any]] = None,
    normalize: bool = True,
    confunder: bool = True,
    num_workers: int = 4,
    fixed_confounder: bool = False,
) -> Dict[str, Any]:
    r"""
    Load the CIFAR-100 dataloaders
    according to what has been specified by the arguments.
    Note that the debug datasets have the confounders by default

    Default:
        batch_size [int] = 128
        test_batch_size [int] = 256
        mean [List[float]] = [0.5071, 0.4867, 0.4408]
        stdev [List[float]] = [0.2675, 0.2565, 0.2761]
        additional_transformations: Optional[List[Any]] = None
        normalize [bool] = True
        confunder [bool] = True
        num_workers [int] = 4
        fixed_confounder [bool] = False

    Args:
        dataset_type [str]: which type of dataset to deploy
        img_size [int]: image shape
        img_depth [int]: depth (number of channels)
        csv_path [str]: path of the images
        test_csv_path [str]: path of the test images
        val_csv_path: [str]: validation path of images
        cifar_metadata [str]: cifar metadata
        device [str]: device
        batch_size [int] = 128
        test_batch_size [int] = 256
        mean [List[float]]: mean
        stdev [List[float]]: stdev
        additional_transformations Optional[List[Any]]: additional train, val and test transform
        normalize [bool]: whether to normalize
        confunder [bool]: whether to put confunders in the images
        num_workers [int]: number of workers of the dataloader
        fixed_confounder [bool] = False: use fixed confounders

    Returns:
        dataloaders [Dict[str, Any]]: a dictionary containing the dataloaders, for training, validation and test
        with different options for confounders and not confounders
    """

    print("#> Loading dataloader ...")

    factory = LoadDatasetFactory()

    # transformations
    transform_train = [
        torchvision.transforms.Resize(img_size),
    ]

    # target transforms
    transform_test = [
        torchvision.transforms.Resize(img_size),
    ]
    hierarchy = cifar_hierarchy

    dataset_train, dataset_validation, dataset_test = None, None, None

    if dataset_type == "mnist":
        hierarchy = mnist_hierarchy

        dataset_train = torchvision.datasets.EMNIST(
            root="./data",
            split="byclass",
            download=True,
            train=True,
        )
        dataset_validation = torchvision.datasets.EMNIST(
            root="./data",
            split="byclass",
            download=True,
            train=True,
        )
        dataset_test = torchvision.datasets.EMNIST(
            root="./data",
            split="byclass",
            download=True,
            train=False,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            dataset_train.data, dataset_train.targets, test_size=0.33, random_state=0
        )
        dataset_train.data = X_train
        dataset_train.targets = y_train
        dataset_validation.data = X_test
        dataset_validation.targets = y_test
        transform_train.extend(
            [
                lambda img: torchvision.transforms.functional.rotate(img, -90),
                torchvision.transforms.RandomHorizontalFlip(p=1),
            ]
        )
        transform_test.extend(
            [
                lambda img: torchvision.transforms.functional.rotate(img, -90),
                torchvision.transforms.RandomHorizontalFlip(p=1),
            ]
        )
    elif dataset_type == "fashion":
        hierarchy = fashion_hierarchy

        dataset_train = torchvision.datasets.FashionMNIST(
            root="./data",
            download=True,
            train=True,
        )
        dataset_validation = torchvision.datasets.FashionMNIST(
            root="./data",
            download=True,
            train=True,
        )
        dataset_test = torchvision.datasets.FashionMNIST(
            root="./data",
            download=True,
            train=False,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            dataset_train.data, dataset_train.targets, test_size=0.33, random_state=0
        )
        dataset_train.data = X_train
        dataset_train.targets = y_train
        dataset_validation.data = X_test
        dataset_validation.targets = y_test

    elif dataset_type == "omniglot":
        hierarchy = omniglot_hierarchy

        dataset_train = torchvision.datasets.Omniglot(
            root="./data",
            download=True,
            background=False,
        )
        dataset_validation = torchvision.datasets.Omniglot(
            root="./data",
            download=True,
            background=False,
        )
        dataset_test = torchvision.datasets.Omniglot(
            root="./data",
            download=True,
            background=False,
        )
        #  (X_train, X_test,) = train_test_split(
        #      dataset_train._flat_character_images, test_size=0.33, random_state=0
        #  )
        #  (
        #      X_train,
        #      X_val,
        #  ) = train_test_split(X_train, test_size=0.2, random_state=0)
        #  dataset_train._flat_character_images = X_train
        #  dataset_validation._flat_character_images = X_val
        #  dataset_test._flat_character_images = X_test

    transform_train.append(torchvision.transforms.ToTensor())
    transform_test.append(torchvision.transforms.ToTensor())

    # Additional transformations
    if additional_transformations:
        transform_train.append(*additional_transformations)
        transform_test.append(*additional_transformations)

    # normalization
    print("Dataset type", dataset_type)
    if normalize:
        if (
            dataset_type == "mnist"
            or dataset_type == "fashion"
            or dataset_type == "omniglot"
        ):
            transform_train.append(
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            )
            transform_test.append(
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            )
        else:
            transform_train.append(torchvision.transforms.Normalize(mean, stdev))
            transform_test.append(torchvision.transforms.Normalize(mean, stdev))

    # compose
    transform_train = torchvision.transforms.Compose(transform_train)
    transform_test = torchvision.transforms.Compose(transform_test)

    # datasets, all of them will have confunders
    # training confunders for validation and train
    # test confunders for test and test with labels
    train_dataset = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_train,
        confund=confunder,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_train,
    )

    train_dataset_no_confounder = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_train,
        confund=False,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_train,
    )

    train_dataset_with_labels_and_confunders_position = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_train,
        confunders_position=True,
        name_labels=True,
        confund=True,  # confund=confunder, # always confounded
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_train,
    )

    val_dataset_with_labels_and_confunders_position = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=val_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=confunder,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_validation,
    )

    train_dataset_with_labels_and_confunders_position_only_conf = (
        factory.instantiateDataset(
            dataset_type,
            image_size=img_size,
            image_depth=img_depth,
            csv_path=csv_path,
            cifar_metafile=cifar_metadata,
            transform=transform_train,
            confunders_position=True,
            name_labels=True,
            confund=True,  # confund=confunder, # always confound
            train=True,
            only_confounders=True,
            fixed_confounder=fixed_confounder,
            dataset=dataset_train,
        )
    )

    train_dataset_with_labels_and_confunders_position_no_conf = (
        factory.instantiateDataset(
            dataset_type,
            image_size=img_size,
            image_depth=img_depth,
            csv_path=csv_path,
            cifar_metafile=cifar_metadata,
            transform=transform_train,
            confunders_position=True,
            name_labels=True,
            confund=True,  # confund=confunder, # always confounded
            train=True,
            only_confounders=False,
            no_confounders=True,
            fixed_confounder=fixed_confounder,
            dataset=dataset_train,
        )
    )

    train_dataset_with_labels = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_train,
        name_labels=True,
        confund=confunder,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_train,
    )

    test_dataset = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
    )

    test_dataset_no_confounder = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confund=False,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
    )

    test_dataset_with_labels_and_confunders_pos = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
    )

    test_dataset_with_labels_and_confunders_pos_only = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=confunder,
        train=False,
        only_confounders=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
    )

    test_dataset_with_labels_and_confunders_pos_only_without_confounders = (
        factory.instantiateDataset(
            dataset_type,
            image_size=img_size,
            image_depth=img_depth,
            csv_path=test_csv_path,
            cifar_metafile=cifar_metadata,
            transform=transform_test,
            confunders_position=True,
            name_labels=True,
            confund=False,
            train=False,
            only_confounders=True,
            fixed_confounder=fixed_confounder,
            dataset=dataset_test,
        )
    )

    test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=False,
        train=True,
        only_confounders=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
    )

    test_dataset_with_labels = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        name_labels=True,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
    )

    val_dataset = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=val_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confund=confunder,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_validation,
    )

    val_dataset_no_confounder = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=val_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
        confund=False,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_validation,
    )

    val_dataset_debug = factory.instantiateDataset(
        dataset_type,
        image_size=img_size,
        image_depth=img_depth,
        csv_path=val_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_train,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_validation,
    )

    # Dataloaders
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    training_loader_no_confounder = torch.utils.data.DataLoader(
        train_dataset_no_confounder,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    training_loader_with_labels_names = torch.utils.data.DataLoader(
        train_dataset_with_labels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader_no_confounder = torch.utils.data.DataLoader(
        test_dataset_no_confounder,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_loader_with_labels_name_confunders_pos = torch.utils.data.DataLoader(
        train_dataset_with_labels_and_confunders_position,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader_with_labels_names = torch.utils.data.DataLoader(
        test_dataset_with_labels,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader_with_labels_and_confunders_pos_only = torch.utils.data.DataLoader(
        test_dataset_with_labels_and_confunders_pos_only,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    val_loader_no_confound = torch.utils.data.DataLoader(
        val_dataset_no_confounder,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader_debug = torch.utils.data.DataLoader(
        val_dataset_debug,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("\t# of training samples: %d" % int(len(train_dataset)))
    print("\t# of test samples: %d" % int(len(test_dataset)))

    # get the Giunchiglia train like dictionary
    train = dotdict({"to_eval": train_dataset.get_to_eval()})
    test = dotdict({"to_eval": test_dataset.get_to_eval()})
    val = dotdict({"to_eval": val_dataset.get_to_eval()})

    # count subclasses
    count_subclasses = 0
    for value in hierarchy.values():
        count_subclasses += sum(1 for _ in value)

    print("\t# of super-classes: %d" % int(len(hierarchy.keys())))
    print("\t# of sub-classes: %d" % int(count_subclasses))

    # define R: adjacency matrix
    R = np.zeros(train_dataset.get_A().shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        train_dataset.get_A()
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(train_dataset.get_A())):
        ancestors = list(
            nx.descendants(g, i)
        )  # here we need to use the function nx.descendants() because in the directed graph
        # the edges have source from the descendant and point towards the ancestor
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    # dictionary of loaders
    dataloaders = {
        "train_loader": training_loader,
        "train_loader_no_confounder": training_loader_no_confounder,
        "train_set": train_dataset,
        "train": train,
        "train_R": R,
        "test_loader": test_loader,
        "test_loader_no_confounder": test_loader_no_confounder,
        "test_loader_with_labels_name": test_loader_with_labels_names,
        "train_loader_debug_mode": train_loader_with_labels_name_confunders_pos,
        "val_loader_debug_mode": val_loader_debug,
        "train_dataset_with_labels_and_confunders_position": train_dataset_with_labels_and_confunders_position,
        "val_dataset_with_labels_and_confunders_position": val_dataset_with_labels_and_confunders_position,
        "train_dataset_with_labels_and_confunders_position_only_conf": train_dataset_with_labels_and_confunders_position_only_conf,
        "train_dataset_with_labels_and_confunders_position_no_conf": train_dataset_with_labels_and_confunders_position_no_conf,
        "training_loader_with_labels_names": training_loader_with_labels_names,
        "test_dataset_with_labels_and_confunders_pos": test_dataset_with_labels_and_confunders_pos,
        "test_loader_with_labels_and_confunders_pos_only": test_loader_with_labels_and_confunders_pos_only,
        "test_dataset_with_labels_and_confunders_pos_only_without_confounders": test_dataset_with_labels_and_confunders_pos_only_without_confounders,
        "test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples": test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples,
        "test_set": test_dataset,
        "test": test,
        "val_set": val_dataset,
        "val_loader": val_loader,
        "val_loader_no_confound": val_loader_no_confound,
        "val": val,
    }

    return dataloaders


def load_old_dataloaders(dataset: str, batch_size: int, device: str) -> Dict[str, Any]:
    """
    Load the old dataloaders:
    Args:
        dataset [str]: dataset name
        batch_size [int]: batch size
        device [str]: device
    Returns:
        dataloaders [Dict[str, Any]]: a dictionary containing the dataloaders, for training, validation and test
    """
    if "others" in dataset:
        train, test = initialize_other_dataset(dataset, datasets)
        val = None
        # here we should get the validation set
        train.to_eval, test.to_eval = torch.tensor(
            train.to_eval, dtype=torch.uint8
        ), torch.tensor(test.to_eval, dtype=torch.bool)
    else:
        train, val, test = initialize_dataset(dataset, datasets)
        train.to_eval, val.to_eval, test.to_eval = (
            torch.tensor(train.to_eval, dtype=torch.bool),
            torch.tensor(val.to_eval, dtype=torch.bool),
            torch.tensor(test.to_eval, dtype=torch.bool),
        )

    # define R
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        train.A
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(train.A)):
        ancestors = list(
            nx.descendants(g, i)
        )  # here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    # Rescale data and impute missing data
    if "others" in dataset:
        scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
            (train.X.astype(float))
        )
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
            np.concatenate((train.X, val.X))
        )
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(
            device
        ), torch.tensor(val.Y).to(device)
    train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(
        device
    ), torch.tensor(train.Y).to(device)
    test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(
        device
    ), torch.tensor(test.Y).to(device)

    # create loadercreate loaders
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if "others" not in dataset:
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
        for (x, y) in zip(val.X, val.Y):
            train_dataset.append((x, y))
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_dataset = {}
        val_loader = {}

    # test dataset
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    # loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    dataloaders = {
        "train_loader": train_loader,
        "train_set": train_dataset,
        "train": train,
        "train_R": R,
        "val_loader": val_loader,
        "val_set": val_dataset,
        "val": val,
        "test_loader": test_loader,
        "test_set": test_dataset,
        "test": test,
    }

    return dataloaders
