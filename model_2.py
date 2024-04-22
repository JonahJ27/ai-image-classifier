import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class FakeNet2(nn.Module):
  def __init__(self):
    '''
    Architecture as follows: 
    3 sets of the following sequence: 
    a 3 x 3 convolutional layer, a ReLU, a BatchNorm2D layer 
    and then a 2 x 2 max pooling layer. We then flatten the 
    model and use a linear layer down to 256 nodes, a ReLU, 
    a BatchNorm1D layer, a dropout layer set to 0.5, another 
    linear layer down to 128 nodes, another ReLU, a linear 
    layer down to a singular node, and we finish with a sigmoid.

    '''
    super().__init__()
    
    self.fake_net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3), 
        nn.ReLU(True), 
        nn.BatchNorm2d(64),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), 
        nn.ReLU(True), 
        nn.BatchNorm2d(num_features=128),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), 
        nn.ReLU(True),
        nn.BatchNorm2d(num_features=256),
        nn.Flatten(),
        nn.Linear(4096, 256),
        nn.ReLU(True),
        nn.BatchNorm1d(num_features=256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
  def forward(self, x): 
    """Performs forward pass

    Arguments
    ---------
    x: Tensor
      Image tensor of shape (B, 3, 32, 32)
    
    Returns
    -------
    Tensor
      Value in the range (0, 1) for shape (B, 1)
    """
    logits = self.fake_net(x)
    return logits