from torch.nn.modules import Module
import torch


class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.step = 1

    def forward(self, x):
        return x + self.step


class MyModelV2(Module):
    def __init__(self, m: Module):
        super().__init__()
        self.step = 1
        self.m = m

    def forward(self, x):
        return x + self.m(self.step)


if __name__ == '__main__':
    tensor = torch.ones(2)
    my_model = MyModel()
    my_model_v2 = MyModelV2(my_model)
    print(my_model(tensor))
    print(my_model_v2(tensor))
