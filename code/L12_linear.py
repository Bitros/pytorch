import torch
from torch.nn import Module, Linear


class LinearExplorer(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4, 1)

    def forward(self, i):
        return self.linear(i)


i = torch.tensor([[1, 3, -1, -2],
                  [-2, -3, -1, 5]]
                 , dtype=torch.float32)

torch.reshape(i, (-1, 1, 4, 2))
o = LinearExplorer()(i)
print(o)
