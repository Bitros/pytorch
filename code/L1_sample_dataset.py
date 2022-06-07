from torchvision.datasets import CIFAR10

train_ds = CIFAR10(root='../data', train=True, download=True)
test_ds = CIFAR10(root='../data', train=False, download=True)
img, class_idx = train_ds[0]
print(img.size, train_ds.classes[class_idx])
img.show()

img, class_idx = test_ds[0]
print(img.size, test_ds.classes[class_idx])
img.show()
