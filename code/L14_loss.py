import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

if __name__ == '__main__':
    i = torch.tensor([1, 2, 3], dtype=torch.float)
    o = torch.tensor([1, 3, 5], dtype=torch.float)
    print(i)
    print(o)
    loss = L1Loss()
    result = loss(i, o)
    print(result)
    loss_sum = L1Loss(reduction='sum')
    result = loss_sum(i, o)
    print(result)
    mse_loss = MSELoss()
    result = mse_loss(i, o)
    print(result)

    # CrossEntropyLoss
    cel = CrossEntropyLoss()
    i = torch.randn(3, 3)
    print(i)
    o = torch.randint(5, (3, ))
    # o = torch.empty(3, dtype=torch.long).random_(5)
    print(o)
    result = cel(i, o)
    print(result)

