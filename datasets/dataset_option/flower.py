"""
Flower102 dataset.
Dataset details at: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
"""

import requests
import torch
from tqdm import tqdm
import tarfile
import os
from torch.utils.data import Dataset
import logging
import scipy.io
from PIL import Image
import utils
import numpy as np


urls = {
    'url_images': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
    'url_segmentations': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz',
    'url_img_labels': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
    'url_splits': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat',
}

class Flower102(Dataset):
    def __init__(self, root, split='train', download=False, transform=None,
                 target_transform=None, use_segmentation=False,
                 seg_transform=None):
        self.root = root
        self.data_path = os.path.join(self.root, 'flower102')
        self.split = split
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.use_segmentation = use_segmentation
        self.seg_transform = seg_transform
        self._setup_data()

        self._labels_all = scipy.io.loadmat(
            os.path.join(self.data_path, 'imagelabels.mat'))['labels'].reshape(-1).astype(np.int64) - 1
        setid = {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}[self.split]
        self.setid = scipy.io.loadmat(
            os.path.join(self.data_path, 'setid.mat'))[setid].reshape(-1)

    def _check_files(self):
        files = [
            os.path.exists(os.path.join(self.data_path, os.path.basename(url)))
            for url in urls.values()
        ]
        return all(files)

    def _setup_data(self):
        data_already_downloaded = self._check_files()

        if self.download:
            if data_already_downloaded:
                logging.info('Files already downloaded.')
            else:
                for key, url in urls.items():
                    fname = os.path.basename(url)
                    utils.download_file(url, os.path.join(self.data_path, fname))
                    if fname.endswith('.tgz'):
                        utils.extract_tar(os.path.join(self.data_path, fname))
        else:
            if not data_already_downloaded:
                logging.error('Data not found. You can download it passing "download=True".')

    def __getitem__(self, idx):
        idx_map = self.setid[idx]
        img_path = os.path.join(self.data_path, 'jpg',
                                f'image_{idx_map:05d}.jpg')
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        label = self._labels_all[idx_map - 1]
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.use_segmentation:
            seg_path = os.path.join(self.data_path, 'segmim',
                                    f'segmim_{idx_map:05d}.jpg')
            seg = Image.open(seg_path)
            if self.seg_transform is not None:
                sef = self.seg_transform(seg)
            return img, seg, label
        return img, label

    def __len__(self):
        return len(self.setid)
