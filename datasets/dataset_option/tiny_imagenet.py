##################################################
# Imports
##################################################

from torch.utils.data import Dataset
import subprocess
import os
from PIL import Image


##################################################
# Tiny Imagenet Dataset
##################################################

class TinyImagenetDataset(Dataset):
    """
    Subset of the ImageNet dataset.
    Images are 64x64.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(TinyImagenetDataset, self).__init__()
        self.data_dir = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self._download()
        self.labels_list = self._retrieve_labels_list()
        self.image_paths, self.labels = self._get_data()

    def _download(self):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        if not os.path.exists(f'{self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip'):
            subprocess.run(f'wget -r -nc -P {self.data_dir} {url}'.split())
            subprocess.run(f'unzip -qq -n {self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip -d {self.data_dir}'.split())

    def _retrieve_labels_list(self):
        labels_list = []
        with open(f'{self.data_dir}/tiny-imagenet-200/wnids.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    labels_list += [line]
        return labels_list

    def _get_data(self):
        image_paths, labels = [], []

        # If train
        if self.train:
            for cl_folder in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train')):
                label = self.labels_list.index(cl_folder)
                for image_name in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images')):
                    image_path = f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images/{image_name}'
                    image_paths += [image_path]
                    labels += [label]

        # If validation
        else:
            with open(f'{self.data_dir}/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    image_name, label_str = line.split('\t')[:2]
                    image_path = f'{self.data_dir}/tiny-imagenet-200/val/images/{image_name}'
                    label = self.labels_list.index(label_str)
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
