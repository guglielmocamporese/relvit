"""
Each dataset returns:
    images: torch tensor of shape [C, H, W], the image.
    labels: torch of shape [1], class of the image.
    relations: torch tensor of shape [NUM_PATCHES, NUM_PATCHES], the spatial relations of the patches.
"""

##################################################
# Imports
##################################################

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# custom
from datasets.transforms import get_transforms
from datasets.dataset_option.tiny_imagenet import TinyImagenetDataset
from datasets.dataset_option.flower import Flower102
from datasets.dataset_option.imagenet100 import Imagenet100
from datasets.dataset_option.imagenet import ImageNetDataset
from datasets.dataset_creation.image_dataset import ImageDataset
from datasets.dataset_creation.shuffle_dataset import ShuffleDataset


##################################################
# Datasets
##################################################

def get_datasets(args, transform='default', target_transform='default', patch_transform='default'):
    """
    Return the PyTorch datasets.
    """

    # transform
    transform = get_transforms(args)[0] if transform == 'default' else transform
    target_transform = get_transforms(args)[1] if target_transform == 'default' else target_transform
    patch_transform = get_transforms(args)[2] if patch_transform == 'default' else patch_transform
    
    ds_args = {
        'root': args.data_base_path,
        'download': True,
    }

    ds_functions = {
        'cifar10': CIFAR10,
        'cifar10_14': CIFAR10,
        'cifar10_28': CIFAR10,
        'cifar10_36': CIFAR10,
        'cifar10_56': CIFAR10,
        'cifar10_112': CIFAR10,
        'cifar10_224': CIFAR10,
        'cifar100': CIFAR100,
        'cifar100_224': CIFAR100,
        'tiny_imagenet': TinyImagenetDataset,
        'tiny_imagenet_224': TinyImagenetDataset,
        'flower102': Flower102,
        'flower102_112': Flower102,
        'flower102_56': Flower102,
        'svhn': SVHN,
        'svhn_224': SVHN,
        'imagenet100': Imagenet100,
        'imagenet100_112': Imagenet100,
        'imagenet100_56': Imagenet100,
        'imagenet100_28': Imagenet100,
        'imagenet_subset': ImageNetDataset,
    }
    if args.dataset not in list(ds_functions.keys()):
        raise Exception(f'Error. Dataset {args.dataset} not supported.')

    if args.dataset in ['flower102', 'flower102_112', 'flower102_56']:
        ds_train = Flower102(split='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = Flower102(split='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = Flower102(split='val', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = Flower102(split='test', transform=transform['test'], target_transform=target_transform['test'], **ds_args)


    elif args.dataset in ['svhn', 'svhn_224']:
        ds_train = SVHN(split='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = SVHN(split='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = SVHN(split='test', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None
    
    elif args.dataset == 'imagenet_subset':
        ds_train = ImageNetDataset(root_path=args.data_base_path, partition='train', transform=transform['train'], target_transform=target_transform['train'], labels_path=args.labels_path)
        ds_train_aug = ImageNetDataset(root_path=args.data_base_path, partition='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], labels_path=args.labels_path)
        ds_validation = ImageNetDataset(root_path=args.data_base_path, partition='val', transform=transform['validation'], target_transform=target_transform['validation'], labels_path=args.labels_path)
        ds_test = None

    else:
        ds_train = ds_functions[args.dataset](train=True, transform=transform['train'], 
                                                target_transform=target_transform['train'], **ds_args)
        ds_train_aug = ds_functions[args.dataset](train=True, transform=transform['train_aug'], 
                                                    target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = ds_functions[args.dataset](train=False, transform=transform['validation'], 
                                                    target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        

    # datasets
    if args.task == 'upstream':
        shuffle_dataset_args = {
            'debug': args.debug_dataset,
            'patch_size': args.patch_size,
            'img_size': args.img_height,
            'mega_patches': args.mega_patches,
            'side_megapatches': args.side_megapatches,
            'use_relations': args.use_relations,
            'use_distances': args.use_dist,
            'use_angles': args.use_angle,
            'use_abs_positions': args.use_abs_positions,
        }
        dss = {
            'train': ShuffleDataset(ds_train, patch_transform=patch_transform['train'], **shuffle_dataset_args),
            'train_aug': ShuffleDataset(ds_train_aug, patch_transform=patch_transform['train_aug'], 
                                        shuffle=args.shuffle_patches, **shuffle_dataset_args),
            'validation': ShuffleDataset(ds_validation, patch_transform=patch_transform['validation'], 
                                         **shuffle_dataset_args),
            'test': None,
        }

    elif args.task == 'downstream':
        img_dataset_args = {
            'img_size': args.img_height,
            'patch_size': args.patch_size,
            'use_relations': args.use_relations,
            'use_distances': args.use_dist,
            'use_angles': args.use_angle,
            'use_abs_positions': args.use_abs_positions,
        }
        dss = {
            'train': ImageDataset(ds_train, **img_dataset_args),
            'train_aug': ImageDataset(ds_train_aug, **img_dataset_args),
            'validation': ImageDataset(ds_validation, **img_dataset_args),
            'test': None,
        }

    else:
        raise Exception(f'Error. Task "{args.task}" not supported.')

    return dss


##################################################
# Dataloaders
##################################################

def get_dataloaders(args):
    """
    Return the PyTorch dataloaders.
    """

    # datasets
    transform = 'default'
    target_transform = 'default'
    if args.backbone in ['vit', 't2t_vit', 'swin']:
        patch_transform = 'default'
    else:
        patch_transform = {'train': None, 'train_aug': None, 'validation': None, 'test': None}
    dss = get_datasets(args, transform=transform, target_transform=target_transform, patch_transform=patch_transform)

    # dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    dls = {
        'train': DataLoader(dss['train'], shuffle=False, **dl_args),
        'train_aug': DataLoader(dss['train_aug'], shuffle=True, **dl_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        'test':  None,
    }
    return dls