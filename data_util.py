import os

import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data.dataset import Subset

import constants


augment_transforms = [
            #transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=0.1),
            #transforms.RandomAffine(degrees=180, translate=(0.2, 0.2), shear=(-30,30)),
            transforms.RandomResizedCrop(constants.INPUT_SIZE, scale=(0.5, 1.)),
            transforms.RandomHorizontalFlip(),
        ]

weeds_to_imagenet_transforms = [
            transforms.Resize(constants.INPUT_SIZE),
            transforms.CenterCrop(constants.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #TODO: check this; norm by 255 instead?
        ]

def get_dataloaders(batch_size, augment_data=False, max_num_pts=None):

    train_transforms = weeds_to_imagenet_transforms
    if augment_data: train_transforms = augment_data + train_transforms

    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(weeds_to_imagenet_transforms),
    }

    image_datasets = {x: datasets.ImageFolder(
        os.path.join(constants.DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}

    if max_num_pts:
        train_indices = np.random.choice(len(image_datasets['train']), max_num_pts, replace=False)
        val_indices = np.random.choice(len(image_datasets['val']), max_num_pts, replace=False)
    else:
        train_indices = range(len(image_datasets['train']))
        val_indices = range(len(image_datasets['val']))

    train_set = Subset(image_datasets['train'], train_indices)
    val_set   = Subset(image_datasets['val'], val_indices)
    sets =  {'train' : train_set, 'val' : val_set}

    # Create training and validation dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(
        sets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']}

    return dataloaders


