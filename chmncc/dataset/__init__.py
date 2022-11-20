"""Dataset module
It contains all the classes and methods to deal with the dataset loading together with the preparation of the
original datasets
"""

from .parser import initialize_dataset, initialize_other_dataset
from .datasets import datasets
from .dataloaders import load_old_dataloaders, load_cifar_dataloaders
from .load_cifar import LoadDataset, get_named_label_predictions
