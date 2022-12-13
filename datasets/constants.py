##################################################
# Imports
##################################################

from torchvision import transforms

# custom
from utils import get_dataset_constant 
import datasets.transforms 
from datasets.dataset_option.imagenet import ImageNetDataset


NORMALIZATION = {
    'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)], # [(0.4814, 0.4542, 0.4033), (0.2726, 0.2643, 0.2774)]
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
    'cifar10_14': [(0.4913, 0.4820, 0.4463),(0.2275, 0.2244, 0.2442)],
    'cifar10_28': [(0.4916, 0.4824, 0.4467), (0.2394, 0.2360, 0.2547)],
    'cifar10_36': [(0.4917, 0.4824, 0.4468), (0.2411, 0.2376, 0.2563)],
    'cifar10_56': [(0.4917, 0.4824, 0.4468), (0.2411, 0.2377, 0.2563)],
    'cifar10_112': [(0.4917, 0.4824, 0.4468), (0.2411, 0.2377, 0.2563)],
    'cifar10_224': [(0.4914, 0.4822, 0.4465), (0.2413, 0.2378, 0.2564)],
    'cifar100': [(0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2761)],
    'cifar100_224': [(0.5070, 0.4865, 0.4409), (0.2622, 0.2513, 0.2714)],
    'tiny_imagenet': [(0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)],
    'tiny_imagenet_224': [(0.4805, 0.4484, 0.3978), (0.2667, 0.2584, 0.2722)],
    'svhn': [(0.4378, 0.4439, 0.4729), (0.1981, 0.2011, 0.1970)],
    'svhn_224': [(0.4377, 0.4438, 0.4728), (0.1960, 0.1989, 0.1955)],
    'flower102_56': [(0.5136, 0.4163, 0.3426),(0.2887, 0.2414, 0.2831)],
    'flower102_112': [(0.5115, 0.4160, 0.3410),(0.2951, 0.2488, 0.2882)],
    'imagenet100': [(0.4595, 0.4520, 0.3900), (0.2547, 0.2385, 0.2548)],
    'imagenet100_112': [(0.4583, 0.4505, 0.3889), (0.2512, 0.2347, 0.2510)],
    'imagenet100_56': [(0.4581, 0.4503, 0.3886),(0.2511, 0.2345, 0.2509)],
    'imagenet100_28': [(0.4582, 0.4504, 0.3887),(0.2512, 0.2345, 0.2508)],
}

def get_normalization(dataset, data_base_path = None, labels_path = None):
    normalization = []
    if dataset == 'imagenet_subset':
        if labels_path:
            if labels_path.split('/')[-1] in imagenet_subsets:
                normalization = imagenet_subsets[labels_path.split('/')[-1]]
            else:
                transform = transforms.Compose([
                    datasets.transforms.to_rgb_ifneeded,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    datasets.transforms.gray2rgb_ifneeded,
                    datasets.transforms.rgba2rgb_ifneeded,
                ])
                ds_train = ImageNetDataset(root_path=data_base_path, partition='train', transform=transform, target_transform=None, labels_path=labels_path)
                mean, std = get_dataset_constant(ds_train)
                normalization.append(tuple(mean.tolist()))
                normalization.append(tuple(std.tolist()))
                print(f'Normalization values are: {normalization}')
        else:
            normalization = NORMALIZATION['imagenet']
    else:
        normalization = NORMALIZATION[dataset]
    return normalization