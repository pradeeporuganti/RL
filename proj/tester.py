import torch
import torch.nn as nn

x = torch.tensor([10.])
w = torch.tensor([2.], requires_grad=True)
b = torch.tensor([1000.], requires_grad=True)

z = (w**2)*x + b
z.backward()
print(b.grad)