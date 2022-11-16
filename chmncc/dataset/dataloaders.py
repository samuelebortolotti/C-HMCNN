from typing import Tuple, List, Dict, Any, Optional
import torchvision
import torch
from tqdm import tqdm
from chmncc.dataset import initialize_dataset, initialize_other_dataset, datasets
from chmncc.config.cifar_config import hierarchy
import networkx as nx
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from chmncc.dataset.load_cifar import LoadDataset
from chmncc.utils import dotdict

#### Compute Mean and Stdev ################


def get_mean_std(
    img_size: Tuple[int, int], source_data: str, target_data: str
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    r"""
    Computes mean and standard deviation over the source and
    target dataloader

    This function has been adapted from
    [link]: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py

    Args:

    - img_size [Tuple[int, int]]: image shape
    - source_data [str]: path to the source data
    - target_data [str]: path to the target data

    Returns:
        source mean and stdev, target mean and stdev [Tuple[Tuple[float, float], Tuple[float, float]]
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
    target_data = torchvision.datasets.ImageFolder(target_data, transform=transform)

    # data loader
    source_loader = torch.utils.data.DataLoader(
        dataset=source_data, batch_size=64, shuffle=True
    )
    target_loader = torch.utils.data.DataLoader(
        dataset=target_data, batch_size=64, shuffle=True
    )

    def compute_mean_std(loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        r"""
        Computes mean and standard deviation over the source and
        target dataloader

        Args:

        - loader [torch.utils.data.DataLoader]: dataloader

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

    return (compute_mean_std(source_loader), compute_mean_std(target_loader))


############ Load dataloaders ############


def load_cifar_dataloaders(
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
) -> Dict[str, torch.utils.data.DataLoader]:
    r"""
    Load the CIFAR-100 dataloaders
    according to what has been specified by the arguments

    Default:

    - batch_size [int] = 128
    - test_batch_size [int] = 256
    - mean [List[float]] = [0.5071, 0.4867, 0.4408]
    - stdev [List[float]] = [0.2675, 0.2565, 0.2761]
    - normalize [bool] = True

    Args:

    - img_size: Tuple[int, int]: image shape
    - img_depth: depth
    - csv_path: path of the images
    - cifar_metadata: cifar metadata
    - mean [List[float]]
    - stdev [List[float]]
    - additional_transformations = None
    - normalize [bool] = True

    Returns:
        dataloaders [Dict[str, torch.utils.data.DataLoader]]:
        a dictionary containing the dataloaders, for training and test
    """

    print("#> Loading dataloader ...")

    # transformations
    transform_train = [
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.RandomHorizontalFlip(),
        #  torchvision.transforms.RandomPerspective(distortion_scale=0.2),
        #  torchvision.transforms.ColorJitter(
        #      brightness=0.5, contrast=0.5, saturation=0.5
        #  ),
        torchvision.transforms.ToTensor(),
    ]

    # target transforms
    transform_test = [
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor(),
    ]

    # Additional transformations
    if additional_transformations:
        transform_train.append(*additional_transformations)
        transform_test.append(*additional_transformations)

    # normalization
    if normalize:
        transform_train.append(torchvision.transforms.Normalize(mean, stdev))
        transform_test.append(torchvision.transforms.Normalize(mean, stdev))

    # compose
    transform_train = torchvision.transforms.Compose(transform_train)
    transform_test = torchvision.transforms.Compose(transform_test)

    # datasets
    train_dataset = LoadDataset(
        image_size=img_size,
        image_depth=img_depth,
        csv_path=csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_train,
    )

    test_dataset = LoadDataset(
        image_size=img_size,
        image_depth=img_depth,
        csv_path=test_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
    )

    val_dataset = LoadDataset(
        image_size=img_size,
        image_depth=img_depth,
        csv_path=val_csv_path,
        cifar_metafile=cifar_metadata,
        transform=transform_test,
    )

    # Dataloaders
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
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

    # define R
    R = np.zeros(train_dataset.get_A().shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        train_dataset.get_A()
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(train_dataset.get_A())):
        ancestors = list(
            nx.descendants(g, i)
        )  # here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    # dictionary of loaders
    dataloaders = {
        "train_loader": training_loader,
        "train_set": train_dataset,
        "train": train,
        "train_R": R,
        "test_loader": test_loader,
        "test_set": test_dataset,
        "test": test,
        "val_set": val_dataset,
        "val_loader": val_loader,
        "val": val,
    }

    return dataloaders


def load_old_dataloaders(dataset: str, batch_size: int, device: str):
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
