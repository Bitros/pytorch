import os

import torch
import torchvision
from torch.nn import Module, Linear, Conv2d, Sequential, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class SeqExplorer(Module):
    def __init__(self):
        """
        Sequential module explorer CIFAR10 quick model.
        :return: <img src="Structure-of-CIFAR10-quick-model.png" />
        """
        super().__init__()
        self.seq = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, i):
        return self.seq(i)


if __name__ == '__main__':
    sample = torch.ones(64, 3, 32, 32)
    train_ds = CIFAR10(root='../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    seq = SeqExplorer()
    for images, _ in test_loader:
        output = seq(images)
        print(output.shape)
        print(output)
        exit(0)
