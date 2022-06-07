import torch
from torch.nn import Module, ReLU, Sigmoid


class NonLinearExplorer(Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, i):
        return self.sigmoid(i)


i = torch.tensor([[1, 3, -1, -2],
                  [-2, -3, -1, 5]])

torch.reshape(i, (-1, 1, 4, 2))
o = NonLinearExplorer()(i)
print(o)
