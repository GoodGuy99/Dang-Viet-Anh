import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from source.model.classifier import Classifier, Classifierv2
from source.model.layer import PrintLayer

class BinaryClassifier(Classifier): 
    """ Simple classifier module for detecting whether the steel fold is faulty
    Warning: due to ReLu activation layers, don't set the learning rate higher than 5e-4.
    Lr = 1 to 3 e-4 is recommended"""

    def __init__(self, H, W, in_channels = 1, name = "bin_class"):
        super().__init__(H, W, in_channels, name)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20, 
                               kernel_size = 3, stride = 3, padding = "valid"),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(20, 20, kernel_size=3,stride = 3, padding = "valid"),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(20, 20, kernel_size=3,stride = 3, padding = "valid"),
            nn.ReLU(),            
        )

        # self.flattener defined in parent class

        #number of channels after the Flatten() 
        n_channels = self.get_flatten_channels(H, W)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 48),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

        # Need to specify those before calling step()
        # self.loss_calc = None
        # self.optimizer= None

##########################################################
class BinaryClassifierv2(Classifierv2): 
    """Improvement: 
    + Add resnet18 backbone. Add input attribute 'backbone' is either 'simple' or 'resnet18'
    + Replace ReLu with LeakyReLu slope 0.1 in simple backbone & classifier
    WARNING: If batch_size is too large (>50), the model cannot learn effectively and converges to a constance"""

    def __init__(self, H, W, in_channels = 1, name = "bin_class", backbone = "simple", leaky_relu_slope = 0.1):
      self.leaky_relu_slope = leaky_relu_slope #default to 0.1
      super().__init__(H, W, in_channels, name, backbone)
      
      # self.feature_extractor defined in parent class
      # self.flattener defined in parent class

      # self.feature_extractor.add_module("print_layer", PrintLayer() ) #DEBUG

      #number of channels after the Flatten() 
      n_channels = self.get_flatten_channels(H, W)

      self.classifier = nn.Sequential(
          nn.Linear(n_channels, 96),
          nn.Dropout(0.3),
          nn.ReLU(),

          nn.Linear(96, 48),
          nn.Dropout(0.3),
          nn.ReLU(),

          nn.Linear(48, 1),
          nn.Sigmoid()
      )

      # Need to specify those before calling step()
      # self.loss_calc = None
      # self.optimizer= None

    def _build_simple_f_extractor(self):
      """Override and define its own simple backbone"""
      feature_extractor =  nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=96, 
                               kernel_size = 3, stride = 3, padding = "valid"),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(96, 96, kernel_size=3,stride = 3, padding = "valid"),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(96, 96, kernel_size=3,stride = 3, padding = "valid"),
            nn.ReLU(),            
        )
      return feature_extractor