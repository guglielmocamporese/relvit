##################################################
# Imports
##################################################

import numpy as np
import torch
from torchvision import transforms


##################################################
# Patch Transforms
##################################################

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

class ReplaceNoise(object):
    """
    Replace a patch with random noise
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return torch.randn(tensor.size())*self.std + self.mean

def get_patch_transforms(args):

    if args.patch_trans in ['none', 'None']:
        return None
    
    transformations = args.patch_trans.split('-')
    transformations_setting = {}
    for transformation in args.patch_trans.split('-'):
        try:
            tran, value = transformation.split(':')
            transformations_setting[tran] = float(value)
        except:
            transformations_setting[transformation] = None
    transformations = list(transformations_setting.keys())

    # store transformations 
    patch_tf = [lambda x: x.transpose(0, 1)]
    if 'centCrop' in transformations:
        patch_tf += [transforms.CenterCrop(np.ceil(args.patch_size/1.15)),
                    PerPatchAug(transforms.RandomResizedCrop(args.patch_size,scale=(0.85,0.85),ratio=(1,1)))]
        transformations.remove('centCrop')
    else:
        patch_tf += [PerPatchAug(transforms.RandomCrop(int(np.round(args.patch_size / 1.15)))),
                    transforms.Resize(args.patch_size)]
    if 'colJitter' in transformations:
        prob_application = 0.8
        patch_tf += [PerPatchAug(
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=prob_application))]
        transformations.remove('colJitter')
    if 'grayScale' in transformations:
        prob_application = 0.2
        patch_tf += [PerPatchAug(transforms.RandomGrayscale(p=prob_application))]
        transformations.remove('grayScale')
    if 'randomNoise' in transformations:
        prob_application = 0.2
        patch_tf += [PerPatchAug(
                    transforms.RandomApply([
                        ReplaceNoise()
                    ], p=prob_application))]
        transformations.remove('randomNoise')

    patch_tf += [lambda x: x.transpose(0, 1)]
    if transformations!=[]:
        raise Exception(f'Error. Transformations "{transformations}" not supported.')

    return transforms.Compose(patch_tf)