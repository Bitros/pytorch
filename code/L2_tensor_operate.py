import torch
# operator in Tensor & _TensorBase
t1 = torch.randint(1, 10, (2, 3))
t2 = torch.randint(5, 8, (2, 3))
t3 = torch.randint(1, 10, (3, 4))
print(t1, t2, t3, sep='\n')
print(t1 + t2)
print(t1 % t2)
print(t1 / t2)
# deprecated
print(t1 // t2)
print(t1 ** t2)
print(t1 * t2)
# mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
# print(torch.matmul(t1, t2))
print(t1.matmul(t3))
print(t1.sum())
print(t1.sum(0))
print(t1.sum(1))
