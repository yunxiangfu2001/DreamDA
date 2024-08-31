import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import numpy as np
from PIL import Image
from scipy.io import loadmat

import copy


def make_dataset(root, split_path='train_list.mat'):
    annotations = [i[0][0] for i in loadmat(split_path)['annotation_list']]
    image_ids = [os.path.join(root,'Images',i[0][0]) for i in loadmat(split_path)['file_list']]
    labels = ([i[0]-1 for i in loadmat(split_path)['labels']])
    classes = [i.split('/')[0] for i in annotations]

    class_to_idx = dict(zip(classes,labels))

    return image_ids, labels, classes, class_to_idx



class Dogs(ImageFolder):
    def __init__(self, root, img_dir='stable-diffusionv1.5', train=True, transform=None, target_transform=None, download=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train=train

        unique_classes  = os.listdir(root+'/Annotation')
        class_text = [i.split('-')[1:] for i in unique_classes]
        class_text = ['-'.join(i) if len(i)>1 else i[0] for i in class_text]

        class_text.sort()
        class_to_idx = {class_text[i]: i for i in range(len(class_text))}
        self.class_to_idx = class_to_idx

        # load synthetic data
        image_name = os.listdir(os.path.join(root,img_dir))
        class_names = ['_'.join(i.split('_')[2:]).replace('.jpg','') for i in image_name]

        # labels = [self.class_to_idx[class_name] for class_name in class_names]
        # image_paths = [os.path.join(os.path.join(root,img_dir), i) for i in image_name]

        labels=[]
        image_paths=[]
        for i in range(len(class_names)):
            if class_names[i] in self.class_to_idx.keys():
                labels.append(self.class_to_idx[class_names[i]])
                image_paths.append(os.path.join(os.path.join(root,img_dir), image_name[i]))

        self.image_filename = image_name
        self.labels=labels
        self.class_names = class_names
        self.samples = list(zip(image_paths, labels))



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
    

class Dogs_pseudo_label(ImageFolder):
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
    def __init__(self, root, img_dir='stable-diffusionv1.5', train=True, transform=None, target_transform=None, download=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train=train

        unique_classes  = os.listdir(root+'/Annotation')
        class_text = [i.split('-')[1:] for i in unique_classes]
        class_text = ['-'.join(i) if len(i)>1 else i[0] for i in class_text]

        class_text.sort()
        class_to_idx = {class_text[i]: i for i in range(len(class_text))}
        self.class_to_idx = class_to_idx

        # load synthetic data
        image_name = os.listdir(os.path.join(root,img_dir))
        # remove real samples
        image_name = [i for i in image_name if i[0] != 'n']


        class_names = ['_'.join(i.split('_')[2:]).replace('.jpg','') for i in image_name]

        # labels = [self.class_to_idx[class_name] for class_name in class_names]
        # image_paths = [os.path.join(os.path.join(root,img_dir), i) for i in image_name]

        labels=[]
        image_paths=[]
        for i in range(len(class_names)):
            if class_names[i] in self.class_to_idx.keys():
                labels.append(self.class_to_idx[class_names[i]])
                image_paths.append(os.path.join(os.path.join(root,img_dir), image_name[i]))

        self.image_filename = image_name
        self.labels=labels
        self.class_names = class_names
        self.samples = list(zip(image_paths, labels))

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
            sample_w = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample_w, self.strong_transform(sample)

    def __len__(self):
        return len(self.samples)

class Dogs_caption(ImageFolder):
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
    def __init__(self, root, img_dir='stable-diffusionv1.5', train=True, transform=None, target_transform=None, download=None, loader=default_loader, processor=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train=train
        self.processor = processor

        unique_classes  = os.listdir(root+'/Annotation')
        class_text = [i.split('-')[1:] for i in unique_classes]
        class_text = ['-'.join(i) if len(i)>1 else i[0] for i in class_text]

        class_text.sort()
        class_to_idx = {class_text[i]: i for i in range(len(class_text))}
        self.class_to_idx = class_to_idx

        # load synthetic data
        image_name = os.listdir(os.path.join(root,img_dir))
        class_names = ['_'.join(i.split('_')[2:]).replace('.jpg','') for i in image_name]
        labels = [self.class_to_idx[class_name] for class_name in class_names]

        image_paths = [os.path.join(os.path.join(root,img_dir), i) for i in image_name]
        self.samples = list(zip(image_paths, labels))



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

        return index, sample, self.strong_transform(sample)

    def __len__(self):
        return len(self.samples)
