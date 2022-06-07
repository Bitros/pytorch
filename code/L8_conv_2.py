import torch
import torch.nn.functional as f
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class ConvExplorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    explorer = ConvExplorer()
    for images, targets in dataloader:
        output_data = explorer(images)
        # torch.Size([16, 3, 32, 32])
        print(images.shape)
        # torch.Size([64, 6, 30, 30])
        print(output_data.shape)
