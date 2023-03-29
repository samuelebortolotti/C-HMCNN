"""Pytorch dataset loading script.
Base class, it performs the basic dataset operations plus those ones which are required for the program work
"""

import random
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from chmncc.config import (
    mnist_hierarchy,
    cifar_hierarchy,
    omniglot_hierarchy,
    omniglot_confunders,
    mnist_confunders,
    cifar_confunders,
    fashion_hierarchy,
    fashion_confunders,
    label_confounders,
)
from typing import Any, Dict, Tuple, List, Union
import networkx as nx
import cv2


class LoadDataset(Dataset):
    """Base LoadDataset class"""

    # which node of the hierarchy to skip (root is only a confound)
    to_skip = ["root"]

    # instance variables to specify.

    """Name of the dataset"""
    dataset_type: str
    """Image size"""
    image_size: int
    """Image depth"""
    image_depth: int
    """Whether the label should be returned or not"""
    return_label: bool
    """Whether the confounders position should be returned or not"""
    confunders_position: bool
    """Additional transformations to be applied"""
    transform: Any
    """Whether the name label should be returned or not"""
    name_labels: bool
    """Whether the confounder should be fixed or not"""
    fixed_confounder: bool
    """Whether the confounders should be applied or not"""
    confund: bool
    """Whether the dataset is used in training or in testing phase"""
    train: bool
    """Whether the dataset should contain only the confounded sample classes or not"""
    only_confounders: bool
    """Whether the confounder position should be returned or not"""
    confunders_position: bool
    """The data list"""
    data_list: List
    """The coarse label list"""
    coarse_labels: List
    """The fine label list"""
    fine_labels: List
    """The hierarchy graph"""
    g: nx.Graph
    """List of nodes"""
    nodes: Any
    """Index of the nodes in the graph"""
    nodes_idx: Any
    """Number of superclasses"""
    n_superclasses: int
    """Nodes names without the root node included"""
    nodes_names_without_root: Any
    """What to eval"""
    to_eval: Any
    """Whether the data is the image path or the image tensor"""
    imgs_are_strings: bool
    """Class statistics count"""
    class_count_statistics: Dict[str, int] = {}

    def _no_confounders(
        self, confounders_list: List[Tuple[str, str, str]], phase: str
    ) -> List[Tuple[str, str, str]]:
        """Method which is used to remove the confounded samples from the datalist

        Args:
            confounders_list [List[Tuple[str, str, str]]]: list of images which include the confounded data
            phase [str]: which phase we are in (either train or test)

        Returns:
            new_datalist [List[Tuple[str, str, str]]]: list of data which do not include the confounded samples
        """
        filtered = []

        # get the right confounders
        confunders = cifar_confunders
        if self.dataset_type == "mnist":
            confunders = mnist_confunders
        elif self.dataset_type == "fashion":
            confunders = fashion_confunders
        elif self.dataset_type == "omniglot":
            confunders = omniglot_confunders

        for image, superclass, subclass in confounders_list:
            # check if the sample is confunded
            superclass = superclass.strip()
            subclass = subclass.strip()
            if superclass in confunders:
                for tmp_index in range(len(confunders[superclass][phase])):
                    if confunders[superclass][phase][tmp_index]["subclass"] == subclass:
                        continue
            filtered.append((image, superclass, subclass))
        return filtered

    def _confounders_only(
        self, confounders_list: List[Tuple[str, str, str]], phase: str
    ) -> List[Tuple[str, str, str]]:
        """Method which is used to keep only the confounded images within the list of data

        Args:
            confounders_list [List[Tuple[str, str, str]]]: list of images which include the confounded data
            phase [str]: which phase we are in (either train or test)

        Returns:
            new_datalist [List[Tuple[str, str, str]]]: list of data which include only the confounded samples
        """
        filtered = []

        # get the confounders
        confunders = cifar_confunders
        if self.dataset_type == "mnist":
            confunders = mnist_confunders
        elif self.dataset_type == "fashion":
            confunders = fashion_confunders
        elif self.dataset_type == "omniglot":
            confunders = omniglot_confunders

        print("Filtering confounders only...")
        for image, superclass, subclass in confounders_list:
            # check if the sample is confunded
            superclass = superclass.strip()
            subclass = subclass.strip()
            if superclass in confunders:
                for tmp_index in range(len(confunders[superclass][phase])):
                    if confunders[superclass][phase][tmp_index]["subclass"] == subclass:
                        filtered.append((image, superclass, subclass))
        return filtered

    def _confund(
        self,
        confunder: Dict[str, str],
        seed: int,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, int, int, int, int]:
        """Method used in order to apply the confunders on top of the images
        Which confunders to apply are specified in the confunders.py file in the config directory

        Args:
            confunder [Dict[str, str]]: confunder information, such as the dimension, shape and color
            seed [int]: which seed to use in order to place the confunder
            image [np.ndarray]: image

        Returns:
            image [np.ndarray]: image with the confunder on top
            p0x p0y, [Tuple[int]]: tuple of integers which depicts the staring point where the confunder has been added
            p1x p1y, [Tuple[int]]: tuple of integers which depicts the ending point where the confunder has been added
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
        if not self.fixed_confounder:
            starting_point = 0 if shape == "rectangle" else radius
            # get random point
            x = random.randint(starting_point, self.image_size - crop_width)
            y = random.randint(starting_point, self.image_size - crop_height)
        else:
            x = self.image_size - crop_width
            y = self.image_size - crop_height
        # starting and ending points
        p0 = (x, y)
        p1 = (x + crop_width, y + crop_height)
        if shape == "circle":
            p1 = (int(crop_width / 2),)
        # whether the shape should be filled
        filled = cv2.FILLED if confunder["type"] else 2
        # draw the shape
        if shape == "rectangle":
            cv2.rectangle(image, p0, p1, confunder["color"], filled)

        elif shape == "circle":
            cv2.circle(image, p0, p1[0], confunder["color"], filled)
        else:
            raise Exception("The shape selected does not exist")
        # return the image
        p0x, p0y = p0
        if len(p1) > 1:
            p1x, p1y = p1
        else:
            p1x, p1y = p1[0], p1[0]
        return image, p0x, p0y, p1x, p1y

    def _initializeHierarchicalGraph(
        self,
        hierarchy_name: str,
    ) -> Tuple[
        nx.classes.reportviews.NodeView,
        Dict[nx.classes.reportviews.NodeView, int],
        np.ndarray,
    ]:
        """Init param
        Args:
            hierarchy_name [str]: name of the hierarchy

        Returns:
            nodes [nx.classes.reportviews.NodeView]: nodes of the graph
            nodes_idx [Dict[nx.classes.reportviews.NodeView, int]]: dictionary node - index
            matrix [np.ndarray]: A - matrix representation of the grap else p[0]h
        """

        # get the right hierarchy
        hierarchy = cifar_hierarchy
        if hierarchy_name == "mnist":
            hierarchy = mnist_hierarchy
        elif hierarchy_name == "fashion":
            hierarchy = fashion_hierarchy
        elif hierarchy_name == "omniglot":
            hierarchy = omniglot_hierarchy

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

    def csv_to_list(self) -> List[Tuple[str, str, str]]:
        """Reads the path of the file and its corresponding label
        Returns:
            csv file entries [List[List[str]]]
        """
        data = list()
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

    def _calculate_data_stats(self) -> None:
        """Compute statistics about the data.
        Basically, it populates class_count_statistics dictionary by counting how many samples are associated with those classes
        """
        for idx in range(len(self.data_list)):
            _, superclass, subclass = self.data_list[idx]
            if superclass.strip() not in self.class_count_statistics:
                self.class_count_statistics[superclass] = 0
            if subclass.strip() not in self.class_count_statistics:
                self.class_count_statistics[subclass] = 0
            self.class_count_statistics[subclass.strip()] += 1
            self.class_count_statistics[superclass.strip()] += 1

    def _introduce_inbalance_confounding(self, dataset: str, train: bool) -> None:
        """
        Introduce inbalance confounding for the data.
        The imbalancing is introduced only in train phase
        Args:
            dataset [str]: name of the dataset
            train [bool]: whether the dataset is meant for training or testing
        """

        # nothing to add if the dataset is empty
        if len(label_confounders[dataset]) == 0 or train is False:
            return

        # get the count of samples for each of the data item to confound
        # and get the number of samples to keep according to the weight
        conf_subclasses_number_to_keep: Dict[str, int] = {}
        conf_subclasses_current_counter: Dict[str, int] = {}
        for superclass in label_confounders[dataset].keys():
            for weight, subclass in zip(
                label_confounders[dataset][superclass]["weight"],
                label_confounders[dataset][superclass]["subclasses"],
            ):
                conf_subclasses_number_to_keep[subclass] = int(
                    self.class_count_statistics[subclass] * weight
                )
                conf_subclasses_current_counter[subclass] = 0

        # cut down the samples
        new_datalist: List = list()
        for idx in range(len(self.data_list)):
            _, _, subclass = self.data_list[idx]
            # the number of examples has been already reached, go next
            if (
                subclass in conf_subclasses_number_to_keep
                and conf_subclasses_number_to_keep[subclass]
                > conf_subclasses_current_counter[subclass]
            ):
                continue
            else:
                # append the element
                new_datalist.append(self.data_list[idx])
            # increase the data counter
            if subclass in conf_subclasses_current_counter:
                conf_subclasses_current_counter[subclass] += 1

        # replace the list
        self.data_list = new_datalist

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, str, str, np.ndarray],
        Tuple[np.ndarray, str, str, int, int, int, int, Dict[str, str]],
        Tuple[np.ndarray, np.ndarray, int, int, int, int, Dict[str, str]],
    ]:
        """Returns a single item.
        It adds the confunder if specified in the initialization of the class.

        Args:
            idx [int]: index of the entry
        Returns:
            image [np.ndarray]: image retrieved
            hierarchical_label [np.ndarray]: hierarchical label. This label is basically a 0/1 vector
            with 1s corresponding to the indexes of the nodes which should be predicted. Hence, the dimension
            should be the same as the output layer of the network. This is returned for the standard training

            If label only mode:
                tuple of image [np.ndarray], label_1 [str] and label_2 [str], hierarchical_label[np.ndarray]
            If only confunder position:
                tuple of image [np.ndarray], hierarchical_label [np.ndarray], confunder_pos_1_x [int], confunder_pos_1_y [int], confunder_pos_2_x [int], confunder_pos_2_y [int], confunder_shape[Dict[str, str]]
            If both label and confunder position:
                tuple of image [np.ndarray], superclass[str], subclass[str], confunder_pos_1_x [int], confunder_pos_1_y [int], confunder_pos_2_x [int], confunder_pos_2_y [int], confunder_shape [Dict[str, str]]
            Otherwise:
                tuple of image [np.ndarray], and hierarchical_label[np.ndarray]

            Dict of image [np.ndarray], label_1 [str] and label_2 [str].

            NOTE: Up tp now the dataloader constraints, empty strings and -1 are returned for invalid positions [confunder positions and confunder_shape]
        """

        # get the right confounder
        confunders = cifar_confunders
        if self.dataset_type == "mnist":
            confunders = mnist_confunders
        elif self.dataset_type == "fashion":
            confunders = fashion_confunders
        elif self.dataset_type == "omniglot":
            confunders = omniglot_confunders

        image_path, image, superclass, subclass = None, None, None, None

        if self.return_label:
            image_path, superclass, subclass = self.data_list[idx]
            superclass = superclass.strip()
            subclass = subclass.strip()
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            if self.imgs_are_strings:
                image = cv2.imread(image_path, 0)
            else:
                image = image_path
        else:
            if self.imgs_are_strings:
                image = cv2.imread(image_path)
            else:
                image = image_path
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #  if self.image_size != 32:
        #      cv2.resize(image, (self.image_size, self.image_size))

        # set to null the confudner shape and confunder pos
        confunder_pos_1_x = -1
        confunder_pos_1_y = -1
        confunder_pos_2_x = -1
        confunder_pos_2_y = -1
        confunder_shape = ""

        # Add the confunders
        if self.confund:
            # get the phase
            phase = "train" if self.train else "test"
            if superclass.strip() in confunders:
                # look if the subclass is contained confunder
                confunder_info = filter(
                    lambda x: x["subclass"] == subclass, confunders[superclass][phase]
                )
                for c_shape in confunder_info:
                    ##
                    # PLEASE BE CAREFUL: ONLY THE LAST CONFUNDER IS RETURNED
                    # SO ONLY ONE ELEMENT SHOULD BE IN THE CONFIGURATION LIST FOR TEST OR TRAIN
                    ##
                    # add the confunders to the image and retrieve their position
                    (
                        image,
                        c_pos_1_x,
                        c_pos_1_y,
                        c_pos_2_x,
                        c_pos_2_y,
                    ) = self._confund(c_shape, idx, image)
                    # Extract the position of the confunder
                    confunder_pos_1_x = c_pos_1_x
                    confunder_pos_1_y = c_pos_1_y
                    confunder_pos_2_x = c_pos_2_x
                    confunder_pos_2_y = c_pos_2_y
                    # Get the shape (and all the info) of the confunder
                    confunder_shape = c_shape["shape"]

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

        # requested: labels and confunders
        if self.name_labels and self.confunders_position:
            return (
                image,  # image
                superclass,  # string label
                subclass,  # string label
                hierarchical_label,  # hierarchical label [that matrix of 1 hot encodings]
                confunder_pos_1_x,  # int position
                confunder_pos_1_y,  # int position
                confunder_pos_2_x,  # int position
                confunder_pos_2_y,  # int position
                confunder_shape,  # dictionary containing informations
            )
        elif self.name_labels:  # only the named labels requested
            return (
                image,  # image
                superclass,  # string label
                subclass,  # string label
                hierarchical_label,  # hierarchical label [that matrix of 1 hot encodings]
            )
        elif self.confunders_position:
            return (
                image,  # image
                hierarchical_label,  # hierarchical label [that matrix of 1 hot encodings]
                confunder_pos_1_x,  # integer position
                confunder_pos_1_y,  # integer position
                confunder_pos_2_x,  # integer position
                confunder_pos_2_y,  # integer position
                confunder_shape,  # dictionary containing informations
            )
        else:
            # test dataset with hierarchical labels
            return (image, hierarchical_label)  # image  # matrix of 1 hot encodings


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
    to_skip = ["root"]
    names = []
    for idx in range(len(hierarchical_label)):
        if nodes[idx] not in to_skip and hierarchical_label[idx] > 0.5:
            names.append(nodes[idx])
    return names


def get_named_label_predictions_with_indexes(
    hierarchical_label: torch.Tensor, nodes: List[str]
) -> Dict[int, str]:
    """Retrive the named predictions from the hierarchical ones
    Args:
        hierarchical_label [torch.Tensor]: label prediction
        nodes: List[str]: list of nodes
    Returns:
        named_predictions
    """
    to_skip = ["root"]
    names = {}
    for idx in range(len(hierarchical_label)):
        if nodes[idx] not in to_skip and hierarchical_label[idx] > 0.5:
            names.update({idx: nodes[idx]})
    return names
