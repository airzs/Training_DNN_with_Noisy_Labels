# -*- coding: utf-8 -*-
# Author : zhangsen (1846182474@qq.com)
# Date : 2019-11-04
# Description :

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from utils import Q_matrix

use_cuda = torch.cuda.is_available()


# Modified MNIST dataset by adding index
class MNISTWithIdx(datasets.MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


# Customized dataset with index
class MyDataset(torch.utils.data.Dataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.targets)


# Loading MNIST dataset
def load_mnist(mnist_root='./mnist'):
    train_set = MNISTWithIdx(root=mnist_root, train=True, transform=transforms.ToTensor())
    test_set = MNISTWithIdx(root=mnist_root, train=False, transform=transforms.ToTensor())
    return train_set, test_set


# Synthesising noisy labels by randomly initialized q matrix
def syn_noisy_lables(labels, num_class=10):
    qmatrix = Q_matrix(num_class)
    num_class = qmatrix.shape[0]
    noisy_labels = []

    for l in labels:
        prob = qmatrix[l, :]
        noisy_l = np.random.choice(range(num_class), p=prob)
        noisy_labels.append(noisy_l)

    return torch.Tensor(noisy_labels), qmatrix


# Get data loader of MNIST dataset
def get_dataloader(batch_size=128, noisy=False):
    train_set, test_set = load_mnist()
    if noisy:
        noisy_labels, qmatrix = syn_noisy_lables(train_set.targets)
        train_set.targets = noisy_labels

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# Get data loader of MNIST dataset for bottom-up DNN v1 model
def get_dataloader_bu1(batch_size=128):
    train_set, test_set = load_mnist()
    train_data, train_targets = train_set.data, train_set.targets

    idx_permed = torch.randperm(len(train_set))
    clean, noisy, clean_train = idx_permed[:20000], idx_permed[20000:50000], idx_permed[50000:]

    # Create dataset
    clean_dataset = MyDataset(train_data[clean, :], train_targets[clean], transform=transforms.ToTensor())
    clean_train_dataset = MyDataset(train_data[clean_train, :], train_targets[clean_train],
                                    transform=transforms.ToTensor())
    noisy_dataset = MyDataset(train_data[noisy, :], train_targets[noisy], transform=transforms.ToTensor())

    # Synthesis 30000 clean data to noisy data
    noisy_labels, qmatrix = syn_noisy_lables(train_set.targets[noisy])
    noisy_dataset.targets = torch.Tensor(noisy_labels)

    # Create dataloader
    clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    clean_train_loader = torch.utils.data.DataLoader(
        dataset=clean_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    noisy_loader = torch.utils.data.DataLoader(
        dataset=noisy_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return clean_loader, clean_train_loader, noisy_loader
