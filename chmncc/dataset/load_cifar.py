"""Pytorch dataset loading script.
Implementation taken from https://github.com/Ugenteraan/Deep_Hierarchical_Classification/blob/main/load_dataset.py
"""

import os
import pickle
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from chmncc.config.cifar_config import hierarchy
from chmncc.utils import read_meta
import networkx as nx

# which node of the hierarchy to skip (root is only a confound)
to_skip = ["root"]


class LoadDataset(Dataset):
    """Reads the given csv file and loads the data."""

    def __init__(
        self,
        csv_path,
        cifar_metafile,
        image_size=32,
        image_depth=3,
        return_label=True,
        transform=None,
    ):
        """Init param."""

        assert os.path.exists(csv_path), "The given csv path must be valid!"

        self.csv_path = csv_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.meta_filename = cifar_metafile
        self.transform = transform
        self.data_list = self.csv_to_list()
        self.coarse_labels, self.fine_labels = read_meta(self.meta_filename)

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

    def _initializeHierarchicalGraph(self):
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

    def csv_to_list(self):
        """Reads the path of the file and its corresponding label"""

        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        return data

    def get_to_eval(self):
        return self.to_eval

    def get_A(self):
        return self.A

    def __len__(self):
        """Returns the total amount of data."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Returns a single item."""
        image_path, image, superclass, subclass = None, None, None, None
        if self.return_label:
            image_path, superclass, subclass = self.data_list[idx]
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size != 32:
            cv2.resize(image, (self.image_size, self.image_size))

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # the hierarchical label is compliant with Giunchiglia's model
        # basically, it has all zeros, except for the indexes where there is a parent
        subclass = subclass.strip()
        hierarchical_label = np.zeros(len(self.nodes))
        # set to one all my ancestors
        hierarchical_label[
            [self.nodes_idx.get(a) for a in nx.ancestors(self.g.reverse(), subclass)]
        ] = 1
        # set to one myself
        hierarchical_label[self.nodes_idx[subclass]] = 1

        return image, hierarchical_label

        #  if self.return_label:
        #      return {
        #          "image": image / 255.0,
        #          "label_1": self.coarse_labels.index(superclass.strip(" ")),
        #          "label_2": self.fine_labels.index(subclass.strip(" ")),
        #      }
        #  else:
        #      return {"image": image}