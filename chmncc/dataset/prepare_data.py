def download_file_from_google_drive(id: str, destination: str) -> None:
    """Dowloads a file from a Google Drive link
    Args:
      id (str): shared document id
      destination (str): where to store the dowloaded data
    """
    BASE_URL = "https://drive.google.com/u/1/uc?id="
    gdown.download(f"{BASE_URL}{id}&export=download", destination, quiet=False)


def unzip_file(source: str, dest: str) -> None:
    """Unzip a file in a given destination location
    Args:
      source (str): source zip file
      dest (str): where to store the unzipped data
    """
    with zipfile.ZipFile(source, "r") as zp:
        zp.extractall(dest)


def filter_data(
    classes: List[str], original_dataset: List[str], smaller_dataset: List[str]
) -> None:
    """Filter the full dataset to only preserve the needed classes and considers
    only the `real world` and `product` domain.
    Args:
      classes (List[str]): list of classes to preserve
      original_dataset (List[str]): original datasets domains location
      smaller_dataset (List[str]): smaller datasets domains location
    """
    for d, td in zip(original_dataset, smaller_dataset):
        makedirs(td)
        for c in tqdm(classes):
            c_path = join(d, c)
            c_target = join(td, c)
            copytree(c_path, c_target)
