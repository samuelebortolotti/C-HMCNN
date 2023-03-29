"""Load Omniglot database
"""

import torch
from torch.utils.data import Dataset
from chmncc.dataset.load_dataset import LoadDataset
from typing import Any
from chmncc.config import omniglot_hierarchy
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
        imbalance_dataset: bool = False,
        only_label_confounders: bool = False,
        **kwargs,
    ):
        """Initialization of the Omniglot dataset
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
            img_size: int = 32: image size
        """

        self.dataset_type = "omniglot"
        self.csv_path = ""
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

        # filter for only the label
        if only_label_confounders:
            self.data_list = self._only_label_confounders(self.data_list, "omniglot")

        # calculate statistics on the data
        self._calculate_data_stats()

        if imbalance_dataset:
            self._introduce_inbalance_confounding("omniglot", train)

    def create_hierarchy(self, label: str) -> str:
        """Method which generates the hierarchy out of the data
        Args:
            label [str]: data label
        Returns:
            label [str]: parent label
        """
        return label.split("/", 1)[0]

    def create_sublabel(self, label: str) -> str:
        """Method which generates the label out of the data
        Args:
            label [str]: child label
        Returns:
            label [str]: child label
        """
        return label
