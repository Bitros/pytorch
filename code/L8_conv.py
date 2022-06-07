import torch
import torch.nn.functional as f

input_data = torch.tensor([[1, 2, 0, 3, 1],
                           [0, 1, 2, 3, 1],
                           [1, 2, 1, 0, 0],
                           [5, 2, 3, 1, 1],
                           [2, 1, 0, 1, 1]], dtype=torch.float)
print(input_data.shape)
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]], dtype=torch.float)

input_data = torch.reshape(input_data, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input_data)
print(input_data.shape)
# define weight aka kernel
output_data = f.conv2d(input_data, kernel)
print(output_data)
output_data = f.conv_transpose2d(input_data, kernel)
print(output_data)
print(output_data.shape)
# cannot define weight in this way
conv2d = torch.nn.Conv2d(1, 1, 3, stride=1, dtype=torch.float)
print(conv2d.weight)
output_data = conv2d(input_data)
print(output_data)
