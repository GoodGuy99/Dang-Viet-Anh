import torch.nn as nn
import torch

class PrintLayer(nn.Module):
    """A debug layer that print output of the previous layer"""
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        with torch.no_grad():
          # print(x)
          print(torch.unique(x))

        return x