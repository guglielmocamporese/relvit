##################################################
# Imports
##################################################

from torch.utils.data import Dataset
import subprocess
import os
import glob
import json
from PIL import Image


##################################################
# Imagenet100 Dataset
##################################################

class Imagenet100(Dataset):
    """
    Subset of the ImageNet dataset.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Imagenet100, self).__init__()
        self.data_dir = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self._download()
        self.labels_list = self._retrieve_labels_list()
        self.image_paths, self.labels = self._get_data()

    def _download(self):
        url = 'https://www.kaggle.com/datasets/ambityga/imagenet100/download'
        if not os.path.exists(f'{self.data_dir}/imagenet100/archive.zip'):
            raise Exception(f'Error. Download of Dataset Imagenet100 not yet implemented.' +
                            f'Please download it from {url}')

    def _retrieve_labels_list(self):
        f = open(f'{self.data_dir}/imagenet100/Labels.json', 'r')
        labels_info = json.load(f)
        labels_list = list(labels_info.keys())
        return labels_list

    def _get_data(self):
        image_paths, labels = [], []
    
        # If train
        if self.train:
            train_sets = glob.glob1(f'{self.data_dir}/imagenet100/', 'train*')
            for train_set in train_sets:
                img_folders = os.listdir(f'{self.data_dir}/imagenet100/{train_set}/')
                for folder in img_folders:
                    list_images = os.listdir(f'{self.data_dir}/imagenet100/{train_set}/{folder}/')
                    label = self.labels_list.index(folder)
                    for img in list_images:
                        image_path = f'{self.data_dir}/imagenet100/{train_set}/{folder}/{img}'
                        image_paths += [image_path]
                        labels += [label]

        # If validation
        else:
            val_folders = os.listdir(f'{self.data_dir}/imagenet100/val.X/')
            for folder in val_folders:
                list_images = os.listdir(f'{self.data_dir}/imagenet100/val.X/{folder}/')
                label = self.labels_list.index(folder)
                for img in list_images:
                    image_path = f'{self.data_dir}/imagenet100/val.X/{folder}/{img}'
                    image_paths += [image_path]
                    labels += [label]
        
        return image_paths, labels


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
