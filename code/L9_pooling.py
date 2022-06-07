import torch
import torch.nn.functional as f

input_data = torch.tensor([[1, 2, 0, 3, 1],
                           [0, 1, 2, 3, 1],
                           [1, 2, 1, 0, 0],
                           [5, 2, 3, 1, 1],
                           [2, 1, 0, 1, 1]], dtype=torch.float)
print(input_data.shape)
# if you do not reshape and will raise RuntimeError: non-empty 3D or 4D (batch mode) tensor expected for input
input_data = torch.reshape(input_data, (1, 1, 5, 5))
print(input_data.shape)

# return index of max value in each row
# tensor([[[[2., 3.],
#           [5., 1.]]]]),
# tensor([[[[1,  3],
#           [15, 18]]]])
output_data = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True, return_indices=True)(input_data)
# output_data = f.max_pool2d(input_data, kernel_size=3, ceil_mode=True, return_indices=True)
# output_data = f.max_pool2d(input_data, kernel_size=3, ceil_mode=False)
print(output_data)
