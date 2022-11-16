"""Pre-processing script to read Cifar-100 dataset and write the images onto disk with the corresponding labels recorded in a csv file.
Implementation taken from https://github.com/Ugenteraan/Deep_Hierarchical_Classification/blob/main/process_cifar100.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import imageio
import tarfile
import urllib.request
from tqdm import tqdm
from chmncc.utils import unpickle, read_meta
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from tqdm import tqdm


class Preprocess_Cifar100:
    """Process the pickle files."""

    def __init__(
        self,
        meta_filename: str = "./dataset/pickle_files/meta",
        train_file: str = "./dataset/pickle_files/train",
        test_file: str = "./dataset/pickle_files/test",
        image_write_dir: str = "./dataset/images/",
        csv_write_dir: str = "./dataset/",
        train_csv_filename: str = "train.csv",
        test_csv_filename: str = "test.csv",
    ):
        """Init params.

        Args:
            meta_filename [str]: meta file
            train_file [str]: train file
            test_file [str]: test file
            image_write_dir [str]: directory where to write the directory
            csv_write_dir [str]: csv file directory
            train_csv_filename [str]: train csv file
            test_csv_filename [str]: test csv file name
        """
        self.meta_filename = meta_filename
        self.train_file = train_file
        self.test_file = test_file
        self.image_write_dir = image_write_dir
        self.csv_write_dir = csv_write_dir
        self.train_csv_filename = train_csv_filename
        self.test_csv_filename = test_csv_filename

        if not os.path.exists(self.image_write_dir):
            os.makedirs(self.image_write_dir)

        if not os.path.exists(self.csv_write_dir):
            os.makedirs(self.csv_write_dir)

        self.coarse_label_names, self.fine_label_names = read_meta(
            metafile=self.meta_filename
        )

    def process_data(self, train: bool = True) -> None:
        """Read the train/test data and write the image array and its corresponding label into the disk and a csv file respectively.
        Args:
            train [bool]: whether the data is for training
        """

        if train:
            pickle_file = unpickle(self.train_file)
        else:
            pickle_file = unpickle(self.test_file)

        filenames = [t.decode("utf8") for t in pickle_file[b"filenames"]]
        fine_labels = pickle_file[b"fine_labels"]
        coarse_labels = pickle_file[b"coarse_labels"]
        data = pickle_file[b"data"]

        filenames = [t.decode("utf8") for t in pickle_file[b"filenames"]]
        fine_labels = pickle_file[b"fine_labels"]
        coarse_labels = pickle_file[b"coarse_labels"]
        data = pickle_file[b"data"]

        images = []
        for d in data:
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, :, 0] = np.reshape(d[:1024], (32, 32))
            image[:, :, 1] = np.reshape(d[1024:2048], (32, 32))
            image[:, :, 2] = np.reshape(d[2048:], (32, 32))
            images.append(image)

        if train:
            csv_filename = self.train_csv_filename
        else:
            csv_filename = self.test_csv_filename

        with open(f"{self.csv_write_dir}/{csv_filename}", "w+") as f:
            for i, image in enumerate(images):
                filename = filenames[i]
                coarse_label = self.coarse_label_names[coarse_labels[i]]
                fine_label = self.fine_label_names[fine_labels[i]]
                imageio.imsave(f"{self.image_write_dir}{filename}", image)
                f.write(
                    f"{self.image_write_dir}{filename}, {coarse_label}, {fine_label}\n"
                )


def download_cifar() -> None:
    """Dowloads cifar 100 and decompress it"""
    CIFAR_100 = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    urllib.request.urlretrieve(
        CIFAR_100,
        "cifar-100-python.tar.gz",
    )
    shutil.move("cifar-100-python.tar.gz", "./dataset/")
    file = tarfile.open("./dataset/cifar-100-python.tar.gz")
    file.extractall("./dataset/")
    file.close()
    shutil.move("./dataset/cifar-100-python/test", "./dataset/pickle_files")
    shutil.move("./dataset/cifar-100-python/train", "./dataset/pickle_files")
    shutil.move("./dataset/cifar-100-python/meta", "./dataset/pickle_files")


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the data dowload and preparation
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("dataset", help="Dataset preparation subparser")
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the download and the prepare data script
    Args:
      args (Namespace): command line arguments
    """

    print("\n### Dataset preparation ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # dowload cifar
    print("#> Download Cifar 100...")
    download_cifar()

    print("#> Preprocessing Cifar 100...")

    p = Preprocess_Cifar100()
    p.process_data(train=True)  # process the training set
    p.process_data(train=False)  # process the testing set

    print("Done")
