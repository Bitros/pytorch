import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tensor = torch.zeros(5, 3)
print(tensor)

tensor = torch.ones_like(tensor)
print(tensor)

tensor = torch.ones(5, 3)
print(tensor)

tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]], device=DEVICE)
print(tensor)
print(torch.asarray(tensor))

tensor = torch.eye(4, device=DEVICE)
print(tensor)
print(tensor.stride())
tensor = torch.empty_strided((3, 3), (3, 1), dtype=torch.int8)
print(tensor)

