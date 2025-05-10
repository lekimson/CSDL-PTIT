import torch
import torch.nn as nn

class SoftmaxStable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=0, keepdim=True).values  # Use .values to get the max values
        x_exp = torch.exp(x - x_max)
        partition = x_exp.sum(0, keepdim=True)
        return x_exp / partition

# Testing the code
data = torch.Tensor([1, 2, 3])
softmax_stable = SoftmaxStable()
output = softmax_stable(data)

# Verifying the output
assert round(output[-1].item(), 2) == 0.67
print(output)