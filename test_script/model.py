import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(in_channels=40, out_channels=128, kernel_size=15, bias=False, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, bias=False, dilation=2, padding=2+2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128,128),
            nn.Linear(128, 3)
        )

    # Defining the forward pass    
    def forward(self, x):
        h = self.cnn_layers(x)
        h = h.view(h.size(0), -1)
#         print(x.shape)
        h = self.linear_layers(h)
        return h

if __name__ == "__main__": 
    import pickle


    print(Y_out.shape)