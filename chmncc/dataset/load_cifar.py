"""Pytorch dataset loading script.
Implementation taken from https://github.com/Ugenteraan/Deep_Hierarchical_Classification/blob/main/load_dataset.py
"""

import os
import torch
from chmncc.dataset.load_dataset import LoadDataset
from chmncc.config import cifar_hierarchy
from chmncc.utils import read_meta
from typing import Any
import networkx as nx


class LoadCifar(LoadDataset):
    """Reads the given csv file and loads the data."""

    def __init__(
        self,
        csv_path: str,
        cifar_metafile: str,
        image_size: int = 32,
        image_depth: int = 3,
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
        """Init param

        Args:
            csv_path [str]: cifar CSV
            cifar_metafile [str]: meta file of CIFAR
            image_size [int] = 32: size of the image (same width and height)
            image_depth [int] = 3: number of channels of the images
            return_label [bool] = True: whether to return labels
            transform [Any] = None: torchvision transformation
            name_labels [bool] = whether to use the label name
            confunders_position [bool] = whether to return the confunder position
            only [bool] = whether to return only the confounder data
            confund [bool] = whether to put confunders
            train [bool] = whether the set is training or not (used to apply the confunders)
            fixed_confounder [bool] = False: confounder position is fixed
        """

        assert os.path.exists(csv_path), "The given csv path must be valid!"

        self.csv_path = csv_path
        self.dataset_type = "cifar"
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.confunders_position = confunders_position
        self.meta_filename = cifar_metafile
        self.transform = transform
        self.data_list = self.csv_to_list()
        self.coarse_labels, self.fine_labels = read_meta(self.meta_filename)
        self.name_labels = name_labels
        self.fixed_confounder = fixed_confounder
        self.imgs_are_strings = True

        # check if the hierarchy dictionary is consistent with the csv file
        for k, v in cifar_hierarchy.items():
            if not k in self.coarse_labels:
                print(f"Superclass missing! {k}")
            for subclass in v:
                if not subclass in self.fine_labels:
                    print(f"Subclass missing! {subclass}")

        # compliant with Giunchiglia code
        self.g = nx.DiGraph()
        self.nodes, self.nodes_idx, self.A = self._initializeHierarchicalGraph("cifar")
        self.n_superclasses = len(cifar_hierarchy.items())
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
