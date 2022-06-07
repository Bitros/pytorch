import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
y = (x + 2) ** 2
print(y)

# y = (x+2)^2


# y = [(x1+2)^2 + (x2+2)^2 + (x3+2)^2 + (x4+2)^2]/4
# dy/dx1 = 2(x1+2)/4 = 2(1+2)/4 = 1.5
out = y.mean()
print(out)
# raise call backward twice error
out.backward()
# fix the error but will change result 1.5+6=7.5
# out.backward(retain_graph=True)
print(x.grad)

# y = (x1+2)^2 + (x2+2)^2 + (x3+2)^2 + (x4+2)^2
# dy/dx1 = 2(x1+2) = 2(1+2)=6
out = y.sum()
print(out)
out.backward()
print(x.grad)
