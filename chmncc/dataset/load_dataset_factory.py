from chmncc.dataset.load_cifar import LoadCifar
from chmncc.dataset.load_mnist import LoadMnist
from chmncc.dataset.load_dataset import LoadDataset
from typing import Optional


class LoadDatasetFactory:
    def instantiateDataset(self, dataset_type: str, **kwargs) -> Optional[LoadDataset]:
        if dataset_type == "cifar":
            return LoadCifar(**kwargs)
        elif dataset_type == "mnist":
            return LoadMnist(**kwargs)
        else:
            return None
