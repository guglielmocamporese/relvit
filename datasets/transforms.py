##################################################
# Imports
##################################################

import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torchvision import transforms

# custom
from datasets import constants
from datasets.patch_transforms import get_patch_transforms


##################################################
# Transforms
##################################################

def image2patches(x, patch_size):
    """
    Extract non overlapped patches of a given patch size from a given image.
    """
    C, H, W = x.shape
    if (not isinstance(patch_size, list)) and (not isinstance(patch_size, tuple)):
        patch_size = (patch_size, patch_size)
    x = x.unsqueeze(0) # [1, C, H, W]
    x = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size) # [1, C * prod(PATCH_SIZE), NUM_PATCHES]
    x = x.view(1, C, patch_size[0], patch_size[1], -1).transpose(0, 1) # [C, 1, PATCH_SIZE[0], PATCH_SIZE[1], NUM_PATCHES]
    x = x.transpose(1, -1).squeeze(-1) # [C, NUM_PATCHES, PATCH_SIZE[0], PATCH_SIZE[1]]
    return x

def patches2image(x, output_size):
    C, N_PATCH, H_PATCH, W_PATCH = x.shape
    x = x.unsqueeze(0) # [1, C, N_PATCH, H_PATCH, W_PATCH]
    x = x.reshape(1, C, N_PATCH, -1).transpose(2, 3).reshape(1, -1, N_PATCH) # [1, C, prod(PATCH_SIZE)]
    x = F.fold(x, output_size, kernel_size=(H_PATCH, W_PATCH), stride=(H_PATCH, W_PATCH)) # [1, C, H, W]
    return x.squeeze(0) # [C, H, W]


def compute_label_encoding():
    """
    It assumes 8 classes, from a grid of 3x3 patches.
    
    |  1  |  12 |  2  |  23 |  3  |
    |  14 |  1  |  2  |  3  |  35 |
    |  4  |  4  |  X  |  5  |  5  |   ->   Coded Location Classes Relative to "X".
    |  46 |  6  |  7  |  8  |  85 |
    |  6  |  67 |  7  |  78 |  8  |

    """
    _map_classes = np.array([
        [' 1',  '1',  '2',  '3',  '3'],
        [ '1',  '1',  '2',  '3',  '3'],
        [ '4',  '4', '-1',  '5',  '5'],
        [ '6',  '6',  '7',  '8',  '8'],
        [ '6',  '6',  '7',  '8',  '8'],
    ])
    _kernel = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    labels_matrix = np.zeros([9, 9])
    for i in range(3):
        for j in range(3):
            k = _kernel[i, j]

            _map = _map_classes[2 - i: 5 - i, 2 - j : 5 - j]

            for ii in range(_map.shape[0]):
                for jj in range(_map.shape[1]):
                    cl_iijj = _map[ii, jj]
                    k_iijj = _kernel[ii, jj]
                    labels_matrix[k - 1, k_iijj - 1] = cl_iijj

    # Convert to one-hot vectors
    out = np.zeros([8, 9, 9])
    for i in range(9):
        for j in range(9):
            l = labels_matrix[i, j]
            l = 0 if l == -1 else l
            l = [int(ll) for ll in str(int(l))]
            l = [F.one_hot(torch.tensor(ll) - 1, 8).to(torch.float)
                    if ll != 0 else torch.zeros(8) for ll in l]
            l = torch.stack(l, 0).mean(0).numpy()
            out[:, i, j] = l
    return out

def transform_enc(labels):
    """
    Args:
        labels: tensor of shape [9]
    
    Output:
        labels_matrix: tensor of shape [num_classes, 3, 3]
    """
    enc = compute_label_encoding() # [8, 9, 9]
    enc = torch.tensor(enc)
    enc = enc[:, labels, :][:, :, labels]
    return enc


def smart_resize(x, h, w):
    """
    x: PIL Image.
    """
    h_in, w_in = x.size
    resize = False
    h_out = min(h_in, h)
    w_out = min(w_in, w)
    if h_out < h:
        resize = True
    if w_out < w:
        resize = True
    if resize:
        x = transforms.Resize((h, w))(x)
    return x

def gray2rgb_ifneeded(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

def rgba2rgb_ifneeded(x):
    return x[:3] if x.shape[0] > 3 else x

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

class GaussianMasking:
    def __init__(self, kernel_size, std):
        self.kernel_size = kernel_size
        self.std = std
        self.kernel = gkern(self.kernel_size, self.std)

    def __call__(self, x):
        """
        x: patch of size [c, h, w]
        """
        x_shape = x.shape
        if len(x_shape) == 3:
            kernel = self.kernel.to(x.device).reshape(1, self.kernel_size, self.kernel_size)
        elif len(x_shape) == 4:
            kernel = self.kernel.to(x.device).reshape(1, 1, self.kernel_size, self.kernel_size)
        return x * kernel

def to_rgb_ifneeded(x):
    """
    Given a PIL Image x it returns its rgb converted version, if x is not rgb.
    """
    return x.convert('RGB') if x.mode in ['L', 'RGBA'] else x

class PerPatchAug:
    """
    Transform, per patch (each patch has its own transformation).
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        x_aug = []
        for x_p in x:
            x_aug += [self.fn(x_p)]
        x_aug = torch.stack(x_aug, 0)
        return x_aug

def get_transforms(args):
    """
    Return the transformations for the datasets.
    """

    trans = {}
    trans_target = {}
    patch_trans = {}


    trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
    }
    patch_trans = {
            'train': None,
            'train_aug': get_patch_transforms(args),
            'validation': None,
            'test': None,
    }

    aa_params = dict(
            translate_const=int(args.img_height * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in constants.get_normalization(args.dataset, args.data_base_path, args.labels_path)[0]]),
            Interpolation=Image.BICUBIC,
        )

    trans_end = [
        transforms.ToTensor(),
        gray2rgb_ifneeded,
        rgba2rgb_ifneeded,
        transforms.Normalize(*constants.get_normalization(args.dataset, args.data_base_path, args.labels_path))]

    if args.dataset in ['flower102', 'imagenet100', 'imagenet_subset']:
        trans_train_aug = [
            to_rgb_ifneeded,
            transforms.RandomResizedCrop(args.img_height),
            transforms.RandomHorizontalFlip()]
        trans_train = [
            to_rgb_ifneeded,
            transforms.Resize(256),
            transforms.CenterCrop(args.img_height)]
        trans_validation = [
            to_rgb_ifneeded,
            transforms.Resize(256),
            transforms.CenterCrop(args.img_height)]
        trans_test = [
            to_rgb_ifneeded,
            transforms.Resize(256),
            transforms.CenterCrop(args.img_height)]

    elif args.dataset in ['cifar10_14', 'cifar10_28','cifar10_36', 'cifar10_56', 'cifar10_112', 'cifar10_224', 'cifar100_224', 
    'tiny_imagenet_224','svhn_224','flower102_112','flower102_56', 'imagenet100_112', 'imagenet100_56', 'imagenet100_28']:
        trans_train_aug = [
            to_rgb_ifneeded,
            transforms.RandomResizedCrop(args.img_height),
            transforms.RandomHorizontalFlip()]
        trans_train = [
            to_rgb_ifneeded,
            transforms.Resize((args.img_height,args.img_height))]
        trans_validation = [
            to_rgb_ifneeded,
            transforms.Resize((args.img_height,args.img_height))]
        trans_test = [
            to_rgb_ifneeded,
            transforms.Resize((args.img_height,args.img_height))]

    elif args.dataset in ['cifar10', 'cifar100', 'tiny_imagenet', 'svhn']:
        trans_train_aug = [
            to_rgb_ifneeded,
            transforms.RandomCrop(args.img_height, padding=4),
            transforms.RandomHorizontalFlip()]
        trans_train = [
            to_rgb_ifneeded,
            transforms.Resize(args.img_height)]
        trans_validation = [
            to_rgb_ifneeded,
            transforms.Resize(args.img_height)]
        trans_test = [
            to_rgb_ifneeded,
            transforms.Resize(args.img_height)]

    else:
        print('No transformations match. No transformations are applied to the datasets.')
        trans = {
            'train': None,
            'train_aug': None,
            'validation': None,
            'test': None,
        }

        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
        }

        patch_trans = {
            'train': None,
            'train_aug': None,
            'validation': None,
            'test': None,
        }

        return trans, trans_target, patch_trans
    

    trans = {
        'train_aug': transforms.Compose(trans_train_aug + trans_end),
        'train': transforms.Compose(trans_train + trans_end),
        'validation': transforms.Compose(trans_validation + trans_end),
        'test': transforms.Compose(trans_test + trans_end),
    }

    return trans, trans_target, patch_trans

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def create_megapatches_grid(x, side, patch_size=32):
    """
    Inputs:
        x: image torch tensor of shape [c, h, w].
        side: num_patches along the height (or with).
        patch_size: int, patch size dimension.
    Outputs:
        x: torch tensor of shape [c, side * patch_size, side * patch_size].
    """
    C, H, W = x.shape
    side_in = int(H / patch_size)
    resize = transforms.Resize((patch_size, patch_size))
    idxs_w = torch.randperm(side_in - 1)[:side - 1] + 1 # [side - 1]
    idxs_w = torch.sort(torch.cat([idxs_w, torch.tensor([side_in])]))[0]
    idxs_w = (torch.cat([idxs_w, torch.tensor([0])]) - torch.cat([torch.tensor([0]), idxs_w]))[:-1]
    idxs_w = idxs_w * patch_size
    
    idxs_h = torch.randperm(side_in - 1)[:side - 1] + 1 # [side - 1]
    idxs_h = torch.sort(torch.cat([idxs_h, torch.tensor([side_in])]))[0]
    idxs_h = (torch.cat([idxs_h, torch.tensor([0])]) - torch.cat([torch.tensor([0]), idxs_h]))[:-1]
    idxs_h = idxs_h * patch_size

    x_rows = torch.split(x, list(idxs_h.numpy()), dim=1)
    x_patches = []
    for x_row in x_rows:
        x_cols = torch.split(x_row, list(idxs_h.numpy()), dim=2)
        x_patches_row = []
        for x_patch in x_cols:
            x_patch = resize(x_patch) # [c, patch_size, patch_size]
            x_patches_row += [x_patch]
        x_patches_row = torch.stack(x_patches_row, 1)
        x_patches += [x_patches_row]
    x_patches = torch.stack(x_patches, 1) # [c,side, side, patch_size, patch_size]
    x = patches2image(x_patches.reshape(C, -1, patch_size, patch_size), (side * patch_size, side * patch_size))
    return x