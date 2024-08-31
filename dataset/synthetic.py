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

import copy

class SyntheticDataset(ImageFolder):
    def __init__(self, root, dataset='pet', img_dir='stable-diffusionv1.5', original_data_dir=None,train=True, transform=None, target_transform=None, download=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train=train

        if dataset == 'dog':
            unique_classes  = os.listdir(original_data_dir+'/Annotation')
            class_text = [i.split('-')[1:] for i in unique_classes]
            class_text = ['-'.join(i) if len(i)>1 else i[0] for i in class_text]

            class_text.sort()
            class_to_idx = {class_text[i]: i for i in range(len(class_text))}
            idx_to_class = {i: cls for cls, i in class_to_idx.items()}

        elif dataset == 'car':
            classes = loadmat(str(f'{original_data_dir}/cars_meta.mat'), squeeze_me=True)["class_names"].tolist()
            class_to_idx = {cls.lower().replace('_',' '): i for i, cls in enumerate(classes)}
            idx_to_class = {i: cls for cls, i in class_to_idx.items()}

        elif dataset == 'pet':
            image_ids = []
            _labels = []
            with open(os.path.join(original_data_dir,'annotations/trainval.txt')) as file:
                for line in file:
                    image_id, label, *_ = line.strip().split()
                    image_ids.append(image_id)
                    _labels.append(int(label) - 1)
            classes = [
                " ".join(part.title() for part in raw_cls.split("_"))
                for raw_cls, _ in sorted(
                    {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, _labels)},
                    key=lambda image_id_and_label: image_id_and_label[1],
                )
            ]
            classes = [cls.lower() for cls in  classes]
            class_to_idx = dict(zip(classes, range(len(classes))))
            idx_to_class = {i: cls for cls, i in class_to_idx.items()}

        elif dataset=='caltech101':
            class_text = sorted(os.listdir(root + '/101_ObjectCategories'))
            
            class_to_idx ={}
            for i, cls in enumerate(class_text):
                class_to_idx[cls.replace('_',' ').lower()] = i
            idx_to_class={i: cls for cls, i in class_to_idx.items()}

        elif dataset=='shenzhenTB':
            class_to_idx = {'tuberculosis': 1, 'normal':0}
            idx_to_class={i: cls for cls, i in class_to_idx.items()}
        elif dataset=='imagenet':
            imagenet_classes = pd.read_csv(f'{root}/map_clsloc.txt', delimiter=' ', header=None)
            classid2text = dict(zip(imagenet_classes[0], imagenet_classes[2]))
            class_to_idx = dict(zip(imagenet_classes[0], imagenet_classes[1]))
            class_to_idx = {cls: int(i)-1 for cls, i in class_to_idx.items()}
            idx_to_class = {i: cls for cls, i in class_to_idx.items()}

        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        # load synthetic data
        image_name = os.listdir(os.path.join(root,img_dir))

        # inversion with pets
        if dataset=='pet':
            class_names = [' '.join(i.split('_')[3:-1]).replace('.jpg','').lower()  for i in image_name]
        elif dataset=='caltech101':
            class_names = [' '.join(i.split('_')[3:-2]).replace('.jpg','').lower() for i in image_name]
        elif dataset=='shenzhenTB':
            class_names = [i.split('_')[-1].replace('.jpg','').lower()  for i in image_name]
        elif dataset=='imagenet':
            class_names = [i.split('_')[3].lower() for i in image_name]
    
        else:
            class_names = [' '.join(i.split('_')[3:]).replace('.jpg','').lower() if 'original' not in i else ' '.join(i.split('_')[2:]).replace('.jpg','').lower() for i in image_name]
        
        if dataset == 'car':
            class_names = [i.replace('|', '/') for i in class_names]

         
        labels=[]
        image_paths=[]
        samples_with_invalid_cls=[]
        for i in range(len(class_names)):
            if class_names[i] in self.class_to_idx.keys():
                labels.append(self.class_to_idx[class_names[i]])
                image_paths.append(os.path.join(root,img_dir, image_name[i]))
            else:
                samples_with_invalid_cls.append([image_name[i], class_names[i]])

        print('samples_with_invalid_cls:',len(samples_with_invalid_cls))
        print('example:',samples_with_invalid_cls[:3])

        self.image_filename = image_name
        self.labels=labels
        self.class_names = class_names
        self.samples = list(zip(image_paths, labels))

        print(len(self.samples))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        # sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)