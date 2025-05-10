import torch
import torch.nn as nn

# Softmax thường
class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)

# Softmax ổn định
class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        c = torch.max(x)
        exp_x = torch.exp(x - c)
        return exp_x / torch.sum(exp_x)

data = torch.Tensor([1, 2, 3])

# Softmax thường
softmax = Softmax()
output = softmax(data)
print("Output của Softmax thường:")
print(output)

# Softmax ổn định
softmax_stable = SoftmaxStable()
output = softmax_stable(data)
print("Output của Softmax ổn định:")
print(output)
