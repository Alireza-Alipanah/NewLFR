import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class CifarMnistConvexCombinationDataset(Dataset):
    def __init__(self, mapping, alpha, split="train", transform=None):
        print(len(mapping))
        self.mapping = mapping[mapping["task"] == split]
        print(len(self.mapping))
        self.transform = transform
        self.y_array = self.mapping['y'].values
        self.n_classes = np.unique(self.y_array).size
        self.g_array = self.mapping['g'].values
        self.group_array = self.mapping['g'].values.astype('int')
        self.n_groups = np.unique(self.group_array).size
        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.cifar_filename_array = self.mapping['cifar_img'].values
        self.mnist_filename_array = self.mapping['mnist_img'].values
        self.alpha = alpha
        self.blended_images = []
        for idx in range(len(self.mapping)):
            cifar_img = Image.open(self.cifar_filename_array[idx])
            mnist_img = Image.open(self.mnist_filename_array[idx])
            padding = (2, 2, 2, 2)
            padded_mnist_image = ImageOps.expand(mnist_img, padding, fill=(0)).convert(cifar_img.mode)
            # f alpha is 0.0, a copy of the first image is returned. If alpha is 1.0, a copy of the second image is returned.
            img = Image.blend(cifar_img, padded_mnist_image, self.alpha)
            self.blended_images.append(img)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        img = self.blended_images[idx]
        if self.transform:
            img = self.transform(img)
        return img, y, g, 0