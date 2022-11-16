"""
Original parser
This code was adapted from https://github.com/lucamasera/AWX
"""

import numpy as np
import networkx as nx
import keras
from itertools import chain


# Skip the root nodes
to_skip = ["root", "GO0003674", "GO0005575", "GO0008150"]


class arff_data:
    """All the datasets they provide are in arff, this is the class"""

    def __init__(self, arff_file, is_GO, is_test=False):
        """Initialize the arff_data

        Args:
            arf_file [string]: arff file
            isGO [boolean]: whether it is the GO dataset
            is_test [boolean]: whether the dataset is test
        """
        self.X, self.Y, self.A, self.terms, self.g = parse_arff(
            arff_file=arff_file, is_GO=is_GO, is_test=is_test
        )
        # set all the non-skippable elements
        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)  # compute the mean ignoring the nans
        for i, j in zip(r_, c_):  # set the mean values for the nans
            self.X[i, j] = m[j]


def parse_arff(arff_file, is_GO=False, is_test=False):
    """Parse the arff data

    Args:
        arf_file [string]: arff file
        isGO [boolean]: whether it is the GO dataset
        is_test [boolean]: whether the dataset is test

    Return:
        X [torch.tensor] data instances
        Y [torch.tensor] labels
        R [torch.tensor[torch.tensor]] adjacency
        g [nx.DiGraph] graph
    """
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        # create the graph
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        for num_line, l in enumerate(f):
            if l.startswith("@ATTRIBUTE"):
                if l.startswith("@ATTRIBUTE class"):
                    h = l.split("hierarchical")[1].strip()
                    for branch in h.split(","):
                        terms = branch.split("/")
                        if is_GO:  # GO add edge
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms) == 1:
                                # add edge from root to term
                                g.add_edge(terms[0], "root")
                            else:
                                # create the children terms
                                for i in range(2, len(terms) + 1):
                                    g.add_edge(
                                        ".".join(terms[:i]), ".".join(terms[: i - 1])
                                    )
                    # sort the nodes with respect to the distances of the root
                    nodes = sorted(
                        g.nodes(),
                        key=lambda x: (nx.shortest_path_length(g, x, "root"), x)
                        if is_GO
                        else (len(x.split(".")), x),
                    )
                    # get the nodes list
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    # reverse
                    g_t = g.reverse()
                else:
                    _, f_name, f_type = l.split()

                    if f_type == "numeric" or f_type == "NUMERIC":
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(
                            lambda x, i: [float(x)] if x != "?" else [np.nan]
                        )

                    else:
                        cats = f_type[1:-1].split(",")
                        cats_lens.append(len(cats))
                        d.append(
                            {
                                key: keras.utils.to_categorical(i, len(cats)).tolist()
                                for i, key in enumerate(cats)
                            }
                        )
                        feature_types.append(
                            lambda x, i: d[i].get(x, [0.0] * cats_lens[i])
                        )
            elif l.startswith("@DATA"):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                d_line = l.split("%")[0].strip().split(",")
                lab = d_line[len(feature_types)].strip()

                X.append(
                    list(
                        chain(
                            *[
                                feature_types[i](x, i)
                                for i, x in enumerate(d_line[: len(feature_types)])
                            ]
                        )
                    )
                )

                # build the labels
                for t in lab.split("@"):
                    y_[
                        [
                            nodes_idx.get(a)
                            for a in nx.ancestors(g_t, t.replace("/", "."))
                        ]
                    ] = 1
                    y_[nodes_idx[t.replace("/", ".")]] = 1
                Y.append(y_)
        X = np.array(X)
        Y = np.stack(Y)

    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g


def initialize_dataset(name, datasets):
    """Initialize the dataset

    Args:
        name [str]: name of the dataset to prepare
        datasets [bool, str, str]: whether the dataset is go (?), the train,
        validation and test data location

    Returns:
        train dataset [arff_data]
        validation dataset [arff_data]
        test dataset [arff_data]
    """
    is_GO, train, val, test = datasets[name]
    return arff_data(train, is_GO), arff_data(val, is_GO), arff_data(test, is_GO, True)


def initialize_other_dataset(name, datasets):
    """Initialize the dataset

    Args:
        name [str]: name of the dataset to prepare
        datasets [bool, str, str]: whether the dataset is go (?), the train
        and test data location

    Returns:
        train dataset [arff_data]
        test dataset [arff_data]
    """
    is_GO, train, test = datasets[name]
    return arff_data(train, is_GO), arff_data(test, is_GO, True)
