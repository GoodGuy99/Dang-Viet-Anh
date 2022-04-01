import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from source.model.conv_net import *
import torchvision.models as models

class Classifier(ConvolutionalNetwork): 
    """ Parent class for binary and multi label classifier
    Child class of ConvolutionalNetwork
    
    Includes:
    - flattener module & a way yo get num of flattened channels before classifier
    - Forward function used by all classifiers
    
    NOT includes:
    - Model architecture: feature_extractor + classifier """

    def __init__(self, H, W, in_channels = 1, name = "classifier"):
      """Only flattener is defined with a maxpool 2, and dropout, the rest depends on child's implementation"""
      super().__init__(H, W, name)
      self.in_channels = in_channels #grayscale image => channels = 1

      #Init
      self.feature_extractor = None
      self.flattener = nn.Sequential(
          nn.Dropout2d(0.3),
          nn.MaxPool2d(2,2),
          nn.Flatten(),
      )
      self.classifier = None
        
    def get_flatten_channels(self, H, W):
      """Protected method: Get the number of channels after the Flatten() layer on the run. 
      Method: feed forward an empty tensor to get the shape"""
      n_channels = torch.empty(1, self.in_channels, H, W)
      n_channels = self.feature_extractor(n_channels)
      n_channels = self.flattener(n_channels).size(-1)
      return n_channels

    def forward(self, x):
      """Extract both logits and features after the conv layers"""
      out = {}

      x = self.feature_extractor(x)
      out['features'] = x

      x = self.flattener(x)
      out['logits'] = self.classifier(x) 
            
      return out
################################################################
class Classifierv2(ConvolutionalNetwork): 
    """ Improvement
    - Backbone resnet18 & backbone factory method
    - Placeholder for simple backbone"""

    def __init__(self, H, W, in_channels = 1, name = "classifier", backbone = "simple"):
      """Only flattener is defined with a maxpool 2, and dropout, the rest depends on child's implementation"""
      super().__init__(H, W, name)
      self.in_channels = in_channels #grayscale image => channels = 1
      self.backbone = backbone

      #Init
      self.feature_extractor = self._build_feature_extractor()
      self.flattener = nn.Sequential(
          nn.Dropout2d(0.3),
          nn.MaxPool2d(2,2),
          nn.Flatten(),
      )
      self.classifier = None
        
    def get_flatten_channels(self, H, W):
      """Protected method: Get the number of channels after the Flatten() layer on the run. 
      Method: feed forward an empty tensor to get the shape"""
      n_channels = torch.empty(1, self.in_channels, H, W)
      n_channels = self.feature_extractor(n_channels)
      n_channels = self.flattener(n_channels).size(-1)
      return n_channels

    def forward(self, x):
      """Extract both logits and features after the conv layers"""
      out = {}

      x = self.feature_extractor(x)
      out['features'] = x

      x = self.flattener(x)
      out['logits'] = self.classifier(x) 
            
      return out
    
    def _build_feature_extractor(self):
      """Protected function: choose a backbone for feature extractor like in Factory design pattern"""
      if self.backbone == 'simple':
        return self._build_simple_f_extractor()
      elif self.backbone == 'resnet18':
        return self._build_resnet18_f_extractor()
      else:
        raise Exception("This backbone hasn't been implemented. Please choose 'simple' or 'resnet18'")
        return None

    def _build_simple_f_extractor(self):
      """Abstract placeholder: need to be implemented by children"""
      raise NotImplementedError

    def _build_resnet18_f_extractor(self):
      """Protected function: use modified resnet18 for f extractor"""
      resnet18 = models.resnet18()

      # original definition: self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
      # my case: grayscale image, change in_channels
      resnet18.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

      # Remove 2 last layers: FC channel + AvgAdaptivePool of resnet18
      resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-2])
                                    # ,PrintLayer()
                                    )
      return resnet18 

