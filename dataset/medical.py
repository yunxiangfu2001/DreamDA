import os
from glob import glob
import pandas as pd
import json

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import numpy as np
from PIL import Image
from scipy.io import loadmat

import copy


class ShenzhenTB(ImageFolder):
    def __init__(self, root, split='train', train=True, transform=None, target_transform=None, download=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train=train

        with open(os.path.join(root, f'{split}.json'), 'r') as f:
            data = json.load(f)

        image_paths=[]
        labels=[]
        class_text =[]
        for i in range(len(data)):
            filename = data[i]
            label = filename.split('_')[-1].replace('.png', '')
            if label == '0':
                finding = 'normal'
            elif label == '1':
                finding = 'tuberculosis'
            else:
                continue
            class_text.append(finding)
            image_paths.append(os.path.join(root,'ChinaSet_AllFiles','CXR_png',filename))
            labels.append(int(label))


        class_to_idx = {'tuberculosis': 1, 'normal':0}
        self.class_to_idx = class_to_idx


        self.image_filename = data
        self.labels=labels
        self.class_names = class_text
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