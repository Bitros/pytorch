import os

import torch
import torchvision
from torch.nn import Module, Linear, Conv2d, Sequential, MaxPool2d, Flatten, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class SeqExplorer(Module):
    def __init__(self):
        r"""
         Sequential module explorer CIFAR10 quick model.
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
    train_ds = CIFAR10(root='../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    seq = SeqExplorer()
    loss = CrossEntropyLoss()
    opt = SGD(seq.parameters(), lr=0.01)
    for epoch in range(20):
        running_loss = 0.0
        for images, targets in test_loader:
            outputs = seq(images)
            result_loss = loss(outputs, targets)
            opt.zero_grad()
            result_loss.backward()
            opt.step()
            running_loss = running_loss + result_loss
        print(running_loss)
