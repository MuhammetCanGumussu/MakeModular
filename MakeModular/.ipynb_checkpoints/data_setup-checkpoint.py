"""
Contains functionality for creating Pytorch Dataloaders for
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import requests
import zipfile
from pathlib import Path


NUM_WORKERS = os.cpu_count()

# setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

def walk_through_dir(dir_path):
    """
    walks through dir_path returning its contents.
    Args:
        dir_path (str or pathlib.Path): target directory
    Returns:
        a print out of:
            number of subdirectories in dir_path
            number of images (files) in each subdirectory
            name of each subdirectory
    """
    # current_dir, child_dirs, child_files(leafs)
    for current_node, child_nodes, files in os.walk(dir_walk):
        print(f"There are {len(child_nodes)} directories and {len(files)} images in '{current_node}' .")

def download_data():
    """
    if data is not exists then pizza steak sushi zip file will be downloaded
    and unzipped into data_path
    """
    # if the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
    print(f"{image_path} directory exists.")
    else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

    walk_through_dir(image_path)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """
    Creates training and testing DataLoaders.

    Takes train and test data directory path and creates 
    Pytorch Datasets and DataLoader objects.

    Args:
        train_dir: Path to training data directory.
        test_dir: Path to testing data directory.
        transform: torchvision transforms to perform training and testing data.
        batch_size: Number of samples per batch in dataloader object
        num_workers: Number of worker (process) for dataloader object
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Class names are a list of the target classes.
    
    """
    # Dataset object are created by ImageFolder dataset class (for Image Classification Format)
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    
    # train and test dataloader are created (shuffle is not necessary for testing)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
