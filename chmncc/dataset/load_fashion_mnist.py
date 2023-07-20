"""Pytorch dataset loading script.
"""

import string
import torch
import numpy as np
from torch.utils.data import Dataset
from chmncc.dataset.load_dataset import LoadDataset
from typing import Any
from chmncc.config import fashion_hierarchy
from typing import List
import networkx as nx


class LoadFashionMnist(LoadDataset):
    """Load FashionMnist dataset"""

    # exclude classses
    exclude: List[str] = ["Trouser", "Dress", "Bag"]
    # dresses
    dresses: List[str] = [
        "T-shirt/top",
        "Pullover",
        "Coat",
        "Shirt",
    ]
    # show
    shoes: List[str] = ["Sandal", "Sneaker", "Ankle boot"]

    def __init__(
        self,
        dataset: Dataset,
        return_label: bool = True,
        transform: Any = None,
        name_labels: bool = False,
        confunders_position: bool = False,
        only_confounders: bool = False,
        confund: bool = True,
        train: bool = True,
        no_confounders: bool = False,
        fixed_confounder: bool = False,
        imbalance_dataset: bool = False,
        only_label_confounders: bool = False,
        **kwargs,
    ):
        """Init param for the load fashion MNIST
        Args:
            dataset [Dataset]: emnist dataset
            return_label [bool] = True: whether the label should be returned
            transform [Any] = None: additional transformations
            name_labels [bool] = False: whether to return the label name
            confunders_position [bool] = False: whether to return the confounders position
            only_confounders: [bool] = False: whether only the confounder should be used
            confund [bool] = True: whether the dataset should contain confounders or not
            train [bool] = True: whether the dataset is for training or not
            no_confounders [bool] = False: whether the dataset should contain no confounder
            fixed_confounder [bool] = False: whether the confounders are fixed
            imbalance_dataset [bool] = False: whether to introduce a dataset imbalancing if it is available
            only_label_confounders [bool] = False: whether to return only images from the imbalanced classses
        """

        self.dataset_type = "fashion"
        self.csv_path = ""
        self.image_size = dataset.data.shape[1]
        self.image_depth = 1
        self.return_label = return_label
        self.confunders_position = confunders_position
        self.transform = transform
        self.imgs_are_strings = False

        self.data_list = list()
        for i in range(dataset.data.shape[0]):
            tmp_class = dataset.classes[dataset.targets[i].item()]
            if not self.skip_class(tmp_class):
                self.data_list.append(
                    (
                        dataset.data[i].numpy(),
                        self.create_hierarchy(tmp_class),
                        tmp_class,
                    )
                )

        self.fine_labels = [
            "shoe",
            "dress",
        ]
        self.coarse_labels = self.fine_labels
        self.name_labels = name_labels
        self.fixed_confounder = fixed_confounder

        # compliant with Giunchiglia code
        self.g = nx.DiGraph()
        self.nodes, self.nodes_idx, self.A = self._initializeHierarchicalGraph(
            "fashion"
        )
        self.n_superclasses = len(fashion_hierarchy.items())
        # keep the name of the nodes without the one of the root.
        self.nodes_names_without_root = self.nodes[1:]
        self.to_eval = torch.tensor(
            [t not in self.to_skip for t in self.nodes], dtype=torch.bool
        )

        # set whether to confund
        self.confund = confund
        # whether to have only confounders
        self.only_confounders = only_confounders
        # whether we are in the training phase
        self.train = train
        # whether to have keep the labels confounders only
        self.only_label_confounders = only_label_confounders

        # filter the data according to the confounders
        if only_confounders:
            self.data_list = self._image_confounders_only(
                self.data_list, "train" if self.train else "test"
            )
        elif no_confounders:
            self.data_list = self._no_image_confounders(
                self.data_list, "train" if self.train else "test"
            )

        # calculate statistics on the data
        self._calculate_data_stats()

        if imbalance_dataset:
            self._introduce_inbalance_confounding("fashion", train)

        # filter for only the label
        if only_label_confounders:
            self.data_list = self._only_label_confounders(self.data_list, "fashion")

    def create_hierarchy(self, label: str) -> str:
        """Method which generates the hierarchy out of the data
        Args:
            label [str]: data label
        Returns:
            label [str]: parent label
        """
        if label in self.shoes:
            return "shoe"
        elif label in self.dresses:
            return "dress"
        else:
            return "unknown"

    def skip_class(self, class_name: str) -> bool:
        """Skip class method.
        Args:
            class_name [str]: name of the class
        Returns:
            True if the class needs to be skipped, False otherwise
        """
        if class_name in self.exclude:
            return True
        return False
