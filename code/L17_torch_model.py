import torch
from torch.nn import ReLU
from torchvision.models import vgg16

from L6_nn_module import MyModel

if __name__ == '__main__':
    vgg16 = vgg16(pretrained=False)
    print(vgg16)
    # way 1
    vgg16.add_module('my_model', MyModel())
    print(vgg16)

    # way 2 add
    vgg16.classifier.add_module('my_model', ReLU())
    print(vgg16)
    # way 2 change
    vgg16.classifier[7] = MyModel()
    # way 2 remove
    del(vgg16.classifier[7])

    torch.save(vgg16, './vgg16.pth')
    torch.save(vgg16.state_dict(), './vgg16_sd.pth')

    print(torch.load('./vgg16.pth'))
    print(torch.load('./vgg16_sd.pth'))