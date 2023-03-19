"""Pytorch dataset loading script.
Implementation taken from https://github.com/Ugenteraan/Deep_Hierarchical_Classification/blob/main/load_dataset.py
"""

import string
import torch
import numpy as np
from torch.utils.data import Dataset
from chmncc.dataset.load_dataset import LoadDataset
from typing import Any
from chmncc.config import omniglot_hierarchy
from typing import List
import networkx as nx
from os.path import join


class LoadOmniglot(LoadDataset):
    """Reads the given csv file and loads the data."""

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
        img_size: int = 32,
        **kwargs,
    ):
        """Init param"""

        self.dataset_type = "omniglot"
        self.image_size = img_size
        self.image_depth = 1
        self.return_label = return_label
        self.confunders_position = confunders_position
        self.transform = transform
        self.imgs_are_strings = True

        self.data_list = list()
        for i in range(len(dataset._flat_character_images)):
            image_name, character_class = dataset._flat_character_images[i]
            image_path = join(
                dataset.target_folder, dataset._characters[character_class], image_name
            )
            self.data_list.append(
                (
                    image_path,
                    self.create_hierarchy(dataset._characters[character_class]),
                    self.create_sublabel(dataset._characters[character_class]),
                )
            )

        self.fine_labels = dataset._alphabets
        self.coarse_labels = self.fine_labels
        self.name_labels = name_labels
        self.fixed_confounder = fixed_confounder

        # compliant with Giunchiglia code
        self.g = nx.DiGraph()
        self.nodes, self.nodes_idx, self.A = self._initializeHierarchicalGraph(
            "omniglot"
        )
        self.n_superclasses = len(omniglot_hierarchy.items())
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
        return label.split("/", 1)[0]

    def create_sublabel(self, label: str) -> str:
        return label
