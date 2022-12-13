##################################################
# Imports
##################################################

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import glob
import logging

# Utils
CSV_NAMES = {
    'val': 'LOC_val_solution.csv',
    'test': 'LOC_sample_submission.csv',
}

def parse_labels(root_path, labels_subset_path = None):
    """
    labels are like 'n01440764'
    labels_desc are like ['tench', 'Tinca', 'tinca']
    """
    labels, labels_desc = [], []
    labels_subset = []
    if labels_subset_path:
        with open(labels_subset_path, 'r') as f:
            for line in f:
                for label in line.split():
                    labels_subset.append(label)
        with open(os.path.join(root_path, 'LOC_synset_mapping.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                label = line.split(' ')[0]
                descriptions = [d.replace(',', '').strip() for d in line.split(' ')[1:]]
                if label in labels_subset:
                    labels += [label]
                    labels_desc += [descriptions]
        if len(labels)!=len(labels_subset):
            raise Exception('Provided invalid sublist of Imagenet labels.')
        else:
            logging.info(f'You are working using a subset of {len(labels)} labels from Imagenet.')
    
    else:
        with open(os.path.join(root_path, 'LOC_synset_mapping.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                label = line.split(' ')[0]
                descriptions = [d.replace(',', '').strip() for d in line.split(' ')[1:]]
                labels += [label]
                labels_desc += [descriptions]

    return labels, labels_desc

def get_image_path(root_path, image_id, partition):
    if partition == 'train':
        image_path = os.path.join(root_path, 'ILSVRC/Data/CLS-LOC', partition, image_id.split('_')[0], f'{image_id}.JPEG')
    elif partition == 'val':
        image_path = os.path.join(root_path, 'ILSVRC/Data/CLS-LOC', partition, f'{image_id}.JPEG')
    elif partition == 'test':
        image_path = os.path.join(root_path, 'ILSVRC/Data/CLS-LOC', partition, f'{image_id}.JPEG')
    else:
        raise Exception(f'Error. Partition "{partition}" not supported.')
    return image_path


##################################################
# ImageNet Dataset
##################################################

class ImageNetDataset(Dataset):
    """
    Definition of the ImageNet dataset.
    For using this dataset please download the data at 
    https://www.kaggle.com/c/imagenet-object-localization-challenge/overview.
    The root_path is the path of the unzipped folder of the downloaded file.
    """
    def __init__(self, root_path, partition, transform=None, target_transform=None, labels_path = None):
        super().__init__()
        self.root_path = root_path
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.labels_path = labels_path
        assert self.partition in ['train', 'val', 'test']
        
        self.labels_codes, self.labels_desc = parse_labels(self.root_path, self.labels_path)
        self.image_paths, self.labels = self._parse_image_data()
        
    def _parse_image_data(self):
        if self.partition == 'train':
            labels = []
            image_paths = glob.glob(os.path.join(self.root_path, 'ILSVRC/Data/CLS-LOC/train/*/*'))
            if self.labels_path:
                image_paths = [x for x in image_paths if x.split('/')[-1].split('.')[0].split('_')[0] in self.labels_codes]
            labels = [self.labels_codes.index(x.split('/')[-1].split('.')[0].split('_')[0]) for x in image_paths]
        elif self.partition in ['val', 'test']:
            df = pd.read_csv(os.path.join(self.root_path, CSV_NAMES[self.partition]))
            df['PredictionString'] = df['PredictionString'].map(lambda x: x.split(' ')[0])
            if self.labels_path:
                df = df.loc[df['PredictionString'].isin(self.labels_codes)]
            image_ids = df['ImageId'].values.tolist()
            image_paths = [get_image_path(self.root_path, image_id, self.partition) for image_id in image_ids]
            labels_code = df['PredictionString'].values.tolist()
            labels = [self.labels_codes.index(l) if self.partition in ['train', 'val'] else -1 for l in labels_code]
        return image_paths, labels
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path, y = self.image_paths[idx], self.labels[idx] 
        x = Image.open(image_path)
        
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
