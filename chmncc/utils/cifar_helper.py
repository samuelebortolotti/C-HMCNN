"""Helper functions for the CIFAR100 dataset
Taken from https://github.com/Ugenteraan/Deep_Hierarchical_Classification
"""

import pickle
from typing import Any, Dict, List, Tuple


def unpickle(file: str) -> Dict[str, Any]:
    """Unpickle the given file

    Args:
        file [str]: name of the file to unpickle
    Returns:
        res [Dict[str, Any]]: dictionary of the results
    """

    with open(file, "rb") as f:
        res = pickle.load(f, encoding="bytes")
    return res


def read_meta(metafile: str) -> Tuple[List[str], List[str]]:
    """Read the meta file and return the coarse and fine labels.

    Args:
        metafile [str]: metafile
    Returns:
        coarse label names [List[str]]: superclass label
        file label names [List[str]]: subclass label
    """
    meta_data = unpickle(metafile)
    fine_label_names = [t.decode("utf8") for t in meta_data[b"fine_label_names"]]
    coarse_label_names = [t.decode("utf8") for t in meta_data[b"coarse_label_names"]]
    return coarse_label_names, fine_label_names
