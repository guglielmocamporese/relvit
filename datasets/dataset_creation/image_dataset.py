from torch.utils.data import Dataset
import torch
import math
import torch.nn as nn

# Custom
from datasets.encode_locations import SSLSignal

class ImageDataset(Dataset):
    def __init__(self, ds, img_size, patch_size, use_relations=False, use_distances=False, use_angles=False, 
                 use_abs_positions=False):
        self.ds = ds
        self.img_size = img_size
        self.patch_size = patch_size
        self.ssl_signal = None
        if use_relations or use_distances or use_angles or use_abs_positions:
            self.ssl_signal = SSLSignal(use_relations=use_relations, use_distances=use_distances, 
                                        use_angles=use_angles, use_abs_positions=use_abs_positions)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds.__getitem__(i)
        sample = {
            'images': x,
            'labels': y,
        }
        if self.ssl_signal is not None:
            num_patches = int(self.img_size // self.patch_size) ** 2
            side = int(math.sqrt(num_patches))
            ssl_labels = self.ssl_signal(torch.arange(num_patches).numpy().reshape(side, side))
            sample.update(ssl_labels)
        return sample