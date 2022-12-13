from torch.utils.data import Dataset
import torch
import math
import torch.nn as nn

# Custom
from datasets.encode_locations import SSLSignal
from datasets.transforms import image2patches, patches2image, create_megapatches_grid



class ShuffleDataset(Dataset):
    def __init__(self, image_dataset, shuffle=False, debug=False, patch_size=32, img_size=224, patch_transform=None, 
                 mega_patches=False, side_megapatches=5, use_relations=True, use_distances=True, use_angles=True, 
                 use_abs_positions=True):
        super().__init__()
        self.ds = image_dataset
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_transform = patch_transform
        self.debug = debug
        self.ssl_signal = SSLSignal(use_relations=use_relations, use_distances=use_distances, use_angles=use_angles, 
                                    use_abs_positions=use_abs_positions)
        self.mega_patches = mega_patches
        self.side_megapatches = side_megapatches

        self.num_patches = int(self.img_size // self.patch_size) ** 2 if not self.mega_patches else self.side_megapatches ** 2 
        self.side = int(math.sqrt(self.num_patches))
        if not self.shuffle:
            self.idx_shuffle = torch.arange(self.num_patches)
            self.ssl_labels = self.ssl_signal(self.idx_shuffle.numpy().reshape(self.side, self.side))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = {}
        x, y_cls = self.ds.__getitem__(idx)
        C, H, W = x.shape

        if self.debug:
            sample['images_orig'] = x
            sample['labels_orig'] = y_cls

        # Shuffle patches
        if self.shuffle:
            self.idx_shuffle = torch.randperm(self.num_patches)

        # Mega-patches
        if self.mega_patches:
            x = create_megapatches_grid(x, side=self.side_megapatches, patch_size=self.patch_size)
            _, H, W = x.shape

        if self.patch_transform is not None:
            x = image2patches(x, patch_size=self.patch_size) # [C, N_PATCH, H_PATCH, W_PATCH]
            x = self.patch_transform(x)
            x = x[:, self.idx_shuffle] # [C, N_PATCH, H_PATCH, W_PATCH]
            x = patches2image(x, output_size=(H, W)) # [C, H, W]
        
        if self.shuffle:
            self.ssl_labels = self.ssl_signal(self.idx_shuffle.numpy().reshape(self.side, self.side)) # dict of ssl matrices
        
        sample['images'] = x
        sample['labels'] = y_cls
        sample.update(self.ssl_labels)
        if self.debug:
            sample['idxs_shuffle'] = self.idx_shuffle
        return sample