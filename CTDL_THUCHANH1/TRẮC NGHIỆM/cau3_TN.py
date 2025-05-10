import torch
import torch.nn as nn

class MySoftmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        exp_x = torch.exp(x - torch.max(x))  # Trừ max(x) để tránh overflow
        return exp_x / torch.sum(exp_x)  # Tính softmax

# Test code
data = torch.Tensor([1, 2, 300000000])
my_softmax = MySoftmax()
output = my_softmax(data)
assert round(output[0].item(), 2) == 0.0
print(output)  # tensor([0., 0., nan])