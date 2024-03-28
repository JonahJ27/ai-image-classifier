import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class FakeNet(nn.Module):
  def __init__(self):
    '''
    You need to add ReLU activations after every internal convolution or linear layer.
    Do not add BatchNorm layers

    Hint
    ----
    You want the feature map produced by the final conv layer to be 6x6 
    '''
    super().__init__()
    
    self.fake_net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=1), # To 15 by 15 from 32 by 32
        nn.ReLU(True), 
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2,), # To 15 by 15 from 7 by 7
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # To 7 by 7 from 7 by 7
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3, padding=1), # To 7 by 7 from 7 by 7
        nn.MaxPool2d(kernel_size=3, stride=2), 
        nn.ReLU(True),
        nn.Flatten(),
        nn.Linear(96 * 9, 96 * 4),
        nn.ReLU(True),
        nn.Linear(96 * 4, 96 * 4),
        nn.ReLU(True),
        nn.Linear(96 * 4, 2),
        nn.ReLU(True),
        nn.Softmax(dim=1)
    )
    
  def forward(self, x): 
    """Performs forward pass

    Arguments
    ---------
    x: Tensor
      image tensor of shape (B, 3, 37, 37)
    
    Returns
    -------
    Tensor
      logits (ranging from 0 to 1) tensor with shape (B, 1000)
    """
    logits = self.fake_net(x)
    return logits