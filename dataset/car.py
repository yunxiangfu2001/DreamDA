import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import numpy as np
from PIL import Image
from scipy.io import loadmat
import pandas as pd


def make_dataset(root, split, annotations_path):
    annotations = pd.read_csv(annotations_path,sep=' ', header=None)
    image_ids = list(annotations[0])
    labels = list(annotations[1])
    return image_ids, labels


class Cars(ImageFolder):
    """`Standford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html/>`_ Dataset.
    Args:
        root (string): Root directory path to dataset.
        split (string): dataset split to load. E.g. ``train``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, split, train=True, transform=None, target_transform=None, download=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        #file_names = glob(os.path.join(root, 'cars_' + split))
        if split == 'train':
            annot_file = 'train.txt'
        elif split == 'test':
            annot_file = 'test.txt'
        image_ids, labels  = make_dataset(root, split, os.path.join(root, 'meta', annot_file))
        image_ids = [os.path.join(self.root, 'train',i) for i in image_ids]
        self.samples = list(zip(image_ids, labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)



