import torch
import numpy as np
from torchvision import transforms

data = np.array([
    [[1, 2], [4, 5]],
    [[2, 4], [8, 10]],
], dtype='uint8')
print(data)
print(data.mean(0))
print(data.mean(1))
print(data.mean(2))

print(data.mean(0).sum(0))
print(data.mean(1).sum(0))
print(data.mean(2).sum(0))
print(data.mean(0).sum(1))
print(data.mean(1).sum(1))
print(data.mean(2).sum(1))

channel_mean = torch.zeros(2)
channel_std = torch.zeros(2)
channel_mean += data.mean(2).sum(0)
channel_std += data.std(2).sum(0)
print(channel_mean)
print(channel_std)
