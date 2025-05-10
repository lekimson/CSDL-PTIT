import torch
import torch.nn as nn

class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)  # Tính e^x
        return exp_x / exp_x.sum()  


data = torch.Tensor([5, 2, 4])
my_softmax = MySoftmax()
output = my_softmax(data)

# Kiểm tra kết quả
assert round(output[-1].item(), 2) == 0.26
print(output)