from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

test_data = CIFAR10(root='../data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=4, shuffle=True, drop_last=False)
img, target = test_data[0]
print(img.shape)
print(target)
for images, targets in test_loader:
    # images contain {batch_size} images, each with shape 3x32x32
    print(images.shape)
    print(targets)