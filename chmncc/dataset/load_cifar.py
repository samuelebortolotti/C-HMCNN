"""Pytorch dataset loading script.
Implementation taken from https://github.com/Ugenteraan/Deep_Hierarchical_Classification/blob/main/load_dataset.py
"""

import os
import random
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from chmncc.config import hierarchy, confunders
from chmncc.utils import read_meta
from typing import Any, Dict, Tuple, List, Union
import networkx as nx

# which node of the hierarchy to skip (root is only a confound)
to_skip = ["root"]


class LoadDataset(Dataset):
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
        confund: bool = True,
        train: bool = True,
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
            confund [bool] = whether to put confunders
            train [bool] = whether the set is training or not (used to apply the confunders)
        """

        assert os.path.exists(csv_path), "The given csv path must be valid!"

        self.csv_path = csv_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.meta_filename = cifar_metafile
        self.transform = transform
        self.data_list = self.csv_to_list()
        self.coarse_labels, self.fine_labels = read_meta(self.meta_filename)
        self.name_labels = name_labels

        # check if the hierarchy dictionary is consistent with the csv file
        for k, v in hierarchy.items():
            if not k in self.coarse_labels:
                print(f"Superclass missing! {k}")
            for subclass in v:
                if not subclass in self.fine_labels:
                    print(f"Subclass missing! {subclass}")

        # compliant with Giunchiglia code
        self.g = nx.DiGraph()
        self.nodes, self.nodes_idx, self.A = self._initializeHierarchicalGraph()
        self.to_eval = torch.tensor(
            [t not in to_skip for t in self.nodes], dtype=torch.bool
        )

        # set whether to confund
        self.confund = confund
        # whether we are in the training phase
        self.train = train

    def _confund(
        self,
        confunder: Dict[str, str],
        seed: int,
        image: np.ndarray,
    ) -> np.ndarray:
        """Method used in order to apply the confunders on top of the images
        Which confunders to apply are specified in the confunders.py file in the config directory

        Args:
            confunder [Dict[str, str]]: confunder information, such as the dimension, shape and color
            seed [int]: which seed to use in order to place the confunder
            image [np.ndarray]: image

        Returns:
            image [np.ndarray]: image with the confunder on top
        """
        # the random number generated is the same for the same image over and over
        # in this way the experiment is reproducible
        random.seed(seed)
        # get the random sizes
        crop_width = random.randint(confunder["min_dim"], confunder["max_dim"])
        crop_height = random.randint(confunder["min_dim"], confunder["max_dim"])
        # only for the circle
        radius = int(crop_width / 2)
        # get the shape
        shape = confunder["shape"]
        # generate the segments
        starting_point = 0 if shape == "rectangle" else radius
        # get random point
        x = random.randint(starting_point, 32 - crop_width)
        y = random.randint(starting_point, 32 - crop_height)
        # starting and ending points
        p0 = (x, y)
        p1 = (x + crop_width, y + crop_height)
        # whether the shape should be filled
        filled = cv2.FILLED if confunder["type"] else 2
        # draw the shape
        if shape == "rectangle":
            cv2.rectangle(image, p0, p1, confunder["color"], filled)
        elif shape == "circle":
            cv2.circle(image, p0, int(crop_width / 2), confunder["color"], filled)
        else:
            raise Exception("The shape selected does not exist")
        # return the image
        return image

    def _initializeHierarchicalGraph(
        self,
    ) -> Tuple[
        nx.classes.reportviews.NodeView,
        Dict[nx.classes.reportviews.NodeView, int],
        np.ndarray,
    ]:
        """Init param

        Args:
            csv_path [str]: cifar CSV
            cifar_metafile [str]: meta file of CIFAR
            image_size [int] = 32: size of the image (same width and height)
            image_depth [int] = 3: number of channels of the images
            return_label [bool] = True: whether to return labels
            transform [Any] = None: torchvision transformation

        Returns:
            nodes [nx.classes.reportviews.NodeView]: nodes of the graph
            nodes_idx [Dict[nx.classes.reportviews.NodeView, int]]: dictionary node - index
            matrix [np.ndarray]: A - matrix representation of the graph
        """
        # prepare the hierarchy
        for img_class in hierarchy:
            self.g.add_edge(img_class, "root")
            for sub_class in hierarchy[img_class]:
                self.g.add_edge(sub_class, img_class)

        # get the nodes
        nodes = sorted(
            self.g.nodes(),
            key=lambda x: (nx.shortest_path_length(self.g, x, "root"), x),
        )
        # index of the nodes in the graph
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        return nodes, nodes_idx, np.array(nx.to_numpy_matrix(self.g, nodelist=nodes))

    def csv_to_list(self) -> List[List[str]]:
        """Reads the path of the file and its corresponding label
        Returns:
            csv file entries [List[List[str]]]
        """

        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        return data

    def get_to_eval(self) -> torch.Tensor:
        """Return the entries to eval in a form of a boolean tensor mask [all except to_skip]
        Return:
            to_eval [torch.Tensor]
        """
        return self.to_eval

    def get_A(self) -> np.ndarray:
        """Get A property
        Returns:
            matrix A - matrix representation of the graph [np.ndarray]
        """
        return self.A

    def get_nodes(self) -> List[str]:
        """Get nodes property
        Returns:
            nodes [List[str]]: nodes list
        """
        return self.nodes

    def __len__(self) -> int:
        """Returns the total amount of data.
        Returns:
            number of dataset entries [int]
        """
        return len(self.data_list)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, str, str]]:
        """Returns a single item.
        It adds the confunder if specified in the initialization of the class.

        Args:
            idx [int]: index of the entry
        Returns:
            image [np.ndarray]: image retrieved
            hierarchical_label [np.ndarray]: hierarchical label. This label is basically a 0/1 vector
            with 1s corresponding to the indexes of the nodes which should be predicted. Hence, the dimension
            should be the same as the output layer of the network. This is returned for the standard training

            Dict of image [np.ndarray], label_1 [str] and label_2 [str]. For exploring the dataset
        """
        image_path, image, superclass, subclass = None, None, None, None
        if self.return_label:
            image_path, superclass, subclass = self.data_list[idx]
            superclass = superclass.strip()
            subclass = subclass.strip()
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size != 32:
            cv2.resize(image, (self.image_size, self.image_size))

        # Add the confunders
        if self.confund:
            # get the phase
            phase = "train" if self.train else "test"
            if superclass.strip() in confunders:
                # look if the subclass is contained confunder
                confunder_info = filter(
                    lambda x: x["subclass"] == subclass, confunders[superclass][phase]
                )
                for confunder_shape in confunder_info:
                    # add the confunders
                    image = self._confund(confunder_shape, idx, image)

        # get the PIL image out of it
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # the hierarchical label is compliant with Giunchiglia's model
        # basically, it has all zeros, except for the indexes where there is a parent
        subclass = subclass.strip()
        hierarchical_label = np.zeros(len(self.nodes))
        # set to one all my ancestors including myself
        hierarchical_label[
            [self.nodes_idx.get(a) for a in nx.ancestors(self.g.reverse(), subclass)]
        ] = 1
        # set to one myself
        hierarchical_label[self.nodes_idx[subclass]] = 1

        # dataset containing the real images
        if self.name_labels:
            #  return {
            #      "image": image / 255.0,
            #      "label_1": self.coarse_labels.index(superclass.strip(" ")),
            #      "label_2": self.fine_labels.index(subclass.strip(" ")),
            #  }
            return image, superclass, subclass
        else:
            # test dataset with hierarchical labels
            return image, hierarchical_label


def get_named_label_predictions(
    hierarchical_label: torch.Tensor, nodes: List[str]
) -> List[str]:
    """Retrive the named predictions from the hierarchical ones
    Args:
        hierarchical_label [torch.Tensor]: label prediction
        nodes: List[str]: list of nodes
    Returns:
        named_predictions
    """
    names = []
    for idx in range(len(hierarchical_label)):
        if nodes[idx] not in to_skip and hierarchical_label[idx] > 0.5:
            names.append(nodes[idx])
    return names
