"""config
File which contains the configurations from Giunchiglia et al.
And the configuration for datasets' hierarchies and confounders"""
from .old_config import *
from .label_confounders import label_confounders
from .cifar_config import cifar_hierarchy
from .mnist_hierarchy import mnist_hierarchy
from .fashion_hierarchy import fashion_hierarchy
from .omniglot_hierarchy import omniglot_hierarchy
from .confounders import (
    mnist_confunders,
    cifar_confunders,
    fashion_confunders,
    omniglot_confunders,
)
