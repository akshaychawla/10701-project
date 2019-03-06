from __future__ import print_function
from PIL import Image
import os
import os.path
import glob
import numpy as np
import sys
from torchvision import transforms
from torchvision.datasets import ImageFolder
from scipy.ndimage import imread
import torch.utils.data as data

def get_dataset(root_folder):
    """
    Returns a torch.data.dataset object containing links to the caltech 101
    dataset. Note: Only to be used by get_loader function 
    """
    # Parameters for data loader 
    RESIZE_HEIGHT = 100 
    RESIZE_WIDTH = 100 

    cal_transform = transforms.Compose([
            transforms.Resize((RESIZE_HEIGHT,RESIZE_WIDTH)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten())
    ])

    root_folder = "../101_ObjectCategories/"
    caltech101 = ImageFolder(root=root_folder, transform=cal_transform)

    return caltech101

def get_loader(root_folder, batch_size=16, shuffle=False, num_workers=0, pin_memory=False):
    """
    Returns a data loader for the caltech 101 dataset
    """
    cal101_dset = get_dataset(root_folder) 

    # train test split 
    split_ratio = 0.2 
    dataset_size = len(cal101_dset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(np.floor(split_ratio * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = data.SubsetRandomSampler(train_indices)
    valid_sampler = data.SubsetRandomSampler(val_indices) 

    train_loader = data.DataLoader( cal101_dset, batch_size=batch_size, 
                                    shuffle=shuffle,num_workers=num_workers, sampler=train_sampler, pin_memory=pin_memory)
    validation_loader = data.DataLoader(cal101_dset, batch_size=batch_size,
                                        shuffle=shuffle,num_workers=num_workers, sampler=valid_sampler, pin_memory=pin_memory)

    return train_loader, validation_loader



# import ipdb; ipdb.set_trace()
# train_loader, valid_loader = get_loader("../101_ObjectCategories/")








