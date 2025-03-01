import os
from collections import Counter
import math

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
    

class DSprite(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, spurious_ratio, seed=42, dataset=None, portion_ratio=1, groups=None):
        torch.manual_seed(seed)
        if dataset is None:
            self.ds = np.array(
                np.load(dataset_dir, encoding="latin1", allow_pickle=True)['imgs'])
            default_shape = (2, 3, 6, 40, 32, 32)
            # color, shape, size, orientation, posx, posy
            indices = torch.tensor(np.array(list(range(math.prod(default_shape)))))
    
            self.color_cutoff = math.prod(default_shape[1:])
    
            indices = indices.reshape(default_shape)
            # orientation, color, size, shape, posx, posy
            indices = torch.permute(indices, (3, 0, 2, 1, 4, 5))
    
            # Prune 0 and Pi from orientations, limit the size only to first and last one and shape only to heart:
            all_orientations = torch.arange(40)
            orientation_exclude_indices = [0, 20]
            mask_orientation = torch.ones(
                all_orientations.size(0), dtype=torch.bool)
            mask_orientation[orientation_exclude_indices] = False
    
            all_shapes = torch.arange(6)
            shape_include_indices = [0, 5]
            mask_shape = torch.zeros(all_shapes.size(0), dtype=torch.bool)
            mask_shape[shape_include_indices] = True
    
            indices = indices[mask_orientation, :, :, :, :, :]
            indices = indices[:, :, mask_shape, :, :, :]
            indices = indices[:, :, :, 2:3, :, :]
    
            # Sampling
            indices = indices.reshape(2, indices.shape[0]//2, 2, 2, -1)
            indices = indices.permute(0, 2, 3, 1, 4).reshape(2, 2, -1)
            # print(indices.shape)
            # torch.Size([2, 2, 38912])

            base_density = 1.0
            base_density = 1 / spurious_ratio
            self.groups = {}
            for core in (0, 1):
              for spu in (0, 1):
                  group = indices[core][spu]
                  group_size = len(group)
                  density = (spurious_ratio if spu==core else (1-spurious_ratio))
            
                  sample_size = round(density * base_density * group_size)
                  group_indices = torch.sort(
                      torch.randperm(group_size)[:sample_size]).values
            
                  reduced_group = group[group_indices]
            
                  self.groups[(core, spu)] = reduced_group

        else:
            self.ds = dataset[list(range(round(len(dataset) * portion_ratio)))]
            self.groups = {}
            
            for core in (0, 1):
              for spu in (0, 1):
                  group = groups[(core, spu)]
                  group_size = len(group)
                  density = (spurious_ratio if spu==core else (1-spurious_ratio))
            
                  sample_size = round(portion_ratio * group_size)
            
                  group_indices = torch.sort(
                      torch.randperm(group_size)[:sample_size]).values
            
                  reduced_group = group[group_indices]
                  self.groups[(core, spu)] = reduced_group
            
            default_shape = (2, 3, 40, 32, 32)
            self.color_cutoff = math.prod(default_shape[1:])

    def get_group(self):
        return self.groups

    def get_list(self):
      return self.ds

    def _get_group_index(self, index):
        if index < 0:
            index += len(self)
        for group_label, group in self.groups.items():
            if index >= len(group):
                index -= len(group)
            else:
                return group_label, group[index].item()

    def __len__(self):
        return sum(len(g) for g in self.groups.values())

    def __getitem__(self, index):
        group_label, ds_index = self._get_group_index(index)

        color = (ds_index >= self.color_cutoff)
        if color:
            ds_index -= self.color_cutoff
        img = torch.tensor(self.ds[ds_index], dtype=torch.float32)
        # print(img.shape)
        if color:
            # img = img # turn red
            img = torch.stack([torch.zeros_like(img), torch.zeros_like(img), img], dim=0)
        else:
            # img = img # turn blue
            img = torch.stack([img, torch.zeros_like(img), torch.zeros_like(img)], dim=0)
        return img, group_label[0], group_label[0] * 2 + group_label[1]
    

class DSpriteImageDataset(Dataset):
    def __init__(self, images, labels, groups):
        self.images = torch.tensor(images, dtype=torch.float32)  # Convert to tensor
        self.labels = torch.tensor(labels, dtype=torch.long)     # Convert to tensor
        self.groups = torch.tensor(groups, dtype=torch.float32)  # Convert to tensor
        self.n_classes = len(np.unique(self.labels))
        self.group_array = groups # for compatibility purposes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.groups[index], 0
    

def build_dsprite_dataset(imgs, ys, gs, indexes):
    images = imgs[indexes]
    labels = ys[indexes]
    groups = gs[indexes]
    print('length of set: ' + str(len(indexes)))
    print(Counter(groups))
    return DSpriteImageDataset(images, labels, groups)


def building_dsprite_dataloader(imgs, ys, gs, correlation_num, portion_num, batch_size, shuffle=False):
    if correlation_num == None:
        return DataLoader(build_dsprite_dataset(imgs, ys, gs, np.array(list(range(len(ys))))), batch_size=batch_size, shuffle=shuffle)
        
    correlation_num = int(correlation_num * 100)
    min_indexes = np.where((gs == 1) | (gs == 2))[0]
    maj_indexes = np.where((gs == 0) | (gs == 3))[0]
    
    min_indexes = min_indexes[:round(portion_num * len(min_indexes))]
    maj_indexes = maj_indexes[:round(portion_num * len(maj_indexes))]
    
    min_count = len(min_indexes)
    maj_count = len(maj_indexes)
    
    total = min_count + maj_count
    c_current = (maj_count / total) * 100
    
    if correlation_num == 100:
        return DataLoader(build_dsprite_dataset(imgs, ys, gs, maj_indexes), batch_size=batch_size, shuffle=shuffle)
    if correlation_num == 0:
        return DataLoader(build_dsprite_dataset(imgs, ys, gs, min_indexes), batch_size=batch_size, shuffle=shuffle)
    
    if c_current > correlation_num:
        for i in range(100 - correlation_num):
            if (correlation_num * (min_count - i)) % (100 - correlation_num) == 0:
                a = (correlation_num * (min_count - i)) // (100 - correlation_num)
                b = min_count - i

                min_new = min_indexes[range(b)]
                maj_new = maj_indexes[range(a)]
                indexes = [y for x in [min_new, maj_new] for y in x]
                return DataLoader(build_dsprite_dataset(imgs, ys, gs, indexes), batch_size=batch_size, shuffle=shuffle)
    elif c_current < correlation_num:
        for i in range(correlation_num):
            if ((100 - correlation_num) * (maj_count - i)) % (correlation_num) == 0:
                b = ((100 - correlation_num) * (maj_count - i)) // (correlation_num)
                a = maj_count - i

                min_new = min_indexes[range(b)]
                maj_new = maj_indexes[range(a)]
                indexes = [y for x in [min_new, maj_new] for y in x]
                return DataLoader(build_dsprite_dataset(imgs, ys, gs, indexes), batch_size=batch_size, shuffle=shuffle)
            
    else:
        return DataLoader(build_dsprite_dataset(imgs, ys, gs, np.array(list(range(len(ys))))), batch_size=batch_size, shuffle=shuffle)