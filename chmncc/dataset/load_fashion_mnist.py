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
    """Reads the given csv file and loads the data."""

    exclude: List[str] = ["Trouser", "Dress", "Bag"]
    dresses: List[str] = [
        "T-shirt/top",
        "Pullover",
        "Coat",
        "Shirt",
    ]
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
        **kwargs,
    ):
        """Init param"""

        self.dataset_type = "fashion"
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

        # filter the data according to the confounders
        if only_confounders:
            self.data_list = self._confounders_only(
                self.data_list, "train" if self.train else "test"
            )
        elif no_confounders:
            self.data_list = self._no_confounders(
                self.data_list, "train" if self.train else "test"
            )

    def create_hierarchy(self, label: str) -> str:
        if label in self.shoes:
            return "shoe"
        elif label in self.dresses:
            return "dress"
        else:
            return "unknown"

    def skip_class(self, class_name: str) -> bool:
        if class_name in self.exclude:
            return True
        return False
