"""Pytorch dataset which returns data correlated to the placed confounder"""
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
from chmncc.dataset.load_cifar import LoadDataset
from chmncc.config import (
    label_confounders,
)
import cv2
import numpy as np


class LoadDebugDataset(Dataset):
    """Loads the data from a pre-existing DataLoader
    (it should return the position of the confounder as well as the labels)
    In particular, this class aims to wrap the cifar dataloader with labels and confunder position
    and returns, a part from the label, the confounder mask which is useful for the RRR loss"""

    def __init__(
        self,
        train_set: LoadDataset,
        balance_subclasses: List[str] = [],
        balance_weights: List[float] = [],
    ):
        """Init param: it saves the training set"""
        self.train_set = train_set
        self.balance_subclasses = balance_subclasses
        self.balance_weights = balance_weights

        if len(balance_subclasses) == 0:
            print("Before")
            self.train_set.print_stats()

        # correct the dataset balancing
        self._correct_dataset_balancing()

        if len(balance_subclasses) == 0:
            print("After")
            self.train_set.print_stats()

    def _class_ok(
        self, subclass: str, current: Dict[str, int], target: Dict[str, int]
    ) -> bool:
        return current[subclass] >= target[subclass]

    def _check_end_duplication(
        self, current: Dict[str, int], target: Dict[str, int]
    ) -> bool:
        end: bool = True
        for subclass in current.keys():
            if not self._class_ok(subclass, current, target):
                end = False
                break
        return end

    def _correct_dataset_balancing(self) -> None:
        # nothing to change if the subclasses are empty
        if len(self.balance_subclasses) == 0:
            return

        count_statistics = self.train_set.class_count_statistics

        # get the count of samples for each of the data item to confound
        # and get the number of samples to keep according to the weight
        conf_subclasses_number_to_reach: Dict[str, int] = {}
        conf_subclasses_current_counter: Dict[str, int] = {}
        for idx, subclass in enumerate(self.balance_subclasses):
            conf_subclasses_current_counter[subclass] = count_statistics[subclass]
            conf_subclasses_number_to_reach[subclass] = int(
                count_statistics[subclass] * self.balance_weights[idx]
            )
        print(conf_subclasses_current_counter)
        print(conf_subclasses_number_to_reach)

        # fill the ones which are not present
        for subclass in count_statistics.keys():
            if subclass not in self.balance_subclasses:
                conf_subclasses_current_counter[subclass] = count_statistics[subclass]
                conf_subclasses_number_to_reach[subclass] = count_statistics[subclass]

        # multiply the samples
        new_datalist: List = self.train_set.data_list.copy()
        basic_step: int = 10  # just a random fixed number

        while not self._check_end_duplication(
            conf_subclasses_current_counter, conf_subclasses_number_to_reach
        ):
            for idx in range(len(self.train_set.data_list)):
                _, _, subclass = self.train_set.data_list[idx]

                # go next if the class we are seeing is ok
                if self._class_ok(
                    subclass,
                    conf_subclasses_current_counter,
                    conf_subclasses_number_to_reach,
                ):
                    continue

                # just clone 10 samples at a time at least
                diff = (
                    conf_subclasses_number_to_reach[subclass]
                    - conf_subclasses_current_counter[subclass]
                )
                basic_step = basic_step if diff > basic_step else diff

                for _ in range(basic_step):
                    # append the element
                    new_datalist.append(self.train_set.data_list[idx])
                    conf_subclasses_current_counter[subclass] += 1

        # replace the list
        self.train_set.data_list = new_datalist

        # shuffle the list
        import random

        random.shuffle(self.train_set.data_list)

        # re-calculate the statistics
        self.train_set._calculate_data_stats()

    def __len__(self) -> int:
        """Returns the total amount of data.
        Returns:
            number of dataset entries [int]
        """
        return len(self.train_set)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, str, str]:
        """
        Args:
            idx [int]: index of the elment to be returned

        Returns the data, specifically:
            element: train sample
            hierarchical_label: hierarchical label
            confounder_mask: mask for the RRR loss
            counfounded: whether the sample is confounded or not
            superclass: superclass in string
            subclass: subclass in string

        Note that: the confounder mask is all zero for non-confounded examples
        """
        # get the data from the training_set
        (
            train_sample,
            superclass,
            subclass,
            hierarchical_label,
            confunder_pos1_x,
            confunder_pos1_y,
            confunder_pos2_x,
            confunder_pos2_y,
            confunder_shape,
        ) = self.train_set[idx]

        # parepare the train example and the explainations in the right shape
        single_el = prepare_single_test_sample(single_el=train_sample)

        # compute the confounder mask
        confounder_mask, confounded = compute_mask(
            shape=(train_sample.shape[1], train_sample.shape[2]),
            confunder_pos1_x=confunder_pos1_x,
            confunder_pos1_y=confunder_pos1_y,
            confunder_pos2_x=confunder_pos2_x,
            confunder_pos2_y=confunder_pos2_y,
            confunder_shape=confunder_shape,
        )

        # restore the requires grad flag
        single_el.requires_grad = False

        # returns the data
        return (
            single_el,
            hierarchical_label,
            confounder_mask,
            confounded,
            superclass,
            subclass,
        )


def compute_mask(
    shape: Tuple[int, int],
    confunder_pos1_x: int,
    confunder_pos1_y: int,
    confunder_pos2_x: int,
    confunder_pos2_y: int,
    confunder_shape: str,
) -> Tuple[torch.Tensor, bool]:
    """Compute the mask according to the confounder position and confounder shape

    Args:
        shape [Tuple[int, int]]: shape of the confounder mask
        confuder_pos1_x [int]: x of the starting point
        confuder_pos1_y [int]: y of the starting point
        confuder_pos2_x [int]: x of the ending point
        confuder_pos2_y [int]: y of the ending point
        confunder_shape [Dict[str, Any]]: confunder information

    Returns:
        confounder_mask [torch.Tensor]: tensor highlighting the area where the confounder is present with ones. It is zero elsewhere
        confounded [bool]: whether the sample is confounded or not
    """
    # confounder mask
    confounder_mask = np.zeros(shape)
    # whether the example is confounded
    confounded = True

    if confunder_shape == "rectangle":
        # get the image of the modified gradient
        cv2.rectangle(
            confounder_mask,
            (confunder_pos1_x, confunder_pos1_y),
            (confunder_pos2_x, confunder_pos2_y),
            (255, 255, 255),
            cv2.FILLED,
        )
    elif confunder_shape == "circle":
        # get the image of the modified gradient
        cv2.circle(
            confounder_mask,
            (confunder_pos1_x, confunder_pos1_y),
            confunder_pos2_x,
            (255, 255, 255),
            cv2.FILLED,
        )
    else:
        confounded = False

    # binarize the mask and adjust it the right way
    confounder_mask = torch.tensor((confounder_mask > 0.5).astype(np.float_))

    # return the confounder mask and whether it s confounded
    return confounder_mask, confounded


def prepare_single_test_sample(single_el: torch.Tensor) -> torch.Tensor:
    """Prepare the test sample
    It sets the gradient as required in order to make it compliant with the rest of the algorithm
    Args:
        single_el [torch.Tensor]: sample
    Returns:
        element [torch.Tensor]: single element
    """
    # requires grad
    single_el.requires_grad = True
    # returns element and predictions
    return single_el
