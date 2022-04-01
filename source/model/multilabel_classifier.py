import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from source.model.classifier import Classifier, Classifierv2
import torchvision.models as models
from source.model.layer import PrintLayer

class MultiLabelClassifier(Classifier): 
    """Abstract class that contains the classifier heads for different mul-class"""
    def __init__(self, H, W, in_channels = 1, num_classes = 5, name = "mul_class"):
        super().__init__(H, W, in_channels, name)
        self.num_classes = num_classes

        # self.feature_extractor defined in child class
        # self.flattener defined in parent class
        # self.classifier defined in this class

        # Need to specify those before calling step()
        # self.loss_calc = None
        # self.optimizer = None

    def build_classifier_head(self, n_channels, num_classes):
      """ Common classifier heads for MulClass. Input: number of channels calculated after flattner & number of multilabel classes"""
      return nn.Sequential(
        nn.Linear(n_channels, 96),
        nn.Dropout(0.3),
        nn.ReLU(),

        nn.Linear(96, 48),
        nn.Dropout(0.3),
        nn.ReLU(),

        nn.Linear(48, num_classes),
        nn.Sigmoid()
      )

class MultiLabelClassifier_Backbone_Simple(MultiLabelClassifier): 
  """Simple version of MultiLabelClassifier. Use 3 conv layers as feature extractor"""
  def __init__(self, H, W, in_channels = 1, num_classes = 5, name = "mul_class_simple"):
      super().__init__(H, W, in_channels, num_classes, name)

      self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, 
                               kernel_size = 3, stride = 3, padding = "valid"),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(96, 96, kernel_size=3,stride = 3, padding = "valid"),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(96, 96, kernel_size=3,stride = 3, padding = "valid"),
            nn.ReLU(),            
        )

      #number of channels after the Flatten() 
      n_channels = self.get_flatten_channels(H, W)

      self.classifier = self.build_classifier_head(n_channels, num_classes)

class MultiLabelClassifier_Backbone_ResNet18(MultiLabelClassifier): 
  """Upgraded version of MultiLabelClassifier. Use ResNet18 as feature extractor"""
  def __init__(self, H, W, in_channels = 1, num_classes = 5, name = "mul_class_resnet18"):
      super().__init__(H, W, in_channels, num_classes, name)
      # Feature extractor
      resnet18 = models.resnet18()

      # original definition: self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
      # my case: grayscale image, change in_channels
      resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

      # Remove 2 last layers: FC channel + AvgAdaptivePool of resnet18
      resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-2]))
      self.feature_extractor = resnet18 

      # Number of channels after the Flatten() 
      n_channels = self.get_flatten_channels(H, W)

      self.classifier = self.build_classifier_head(n_channels, num_classes)


####################################VERSION 2
class MultiLabelClassifierv2(Classifierv2): 
    """Integrate backbones into one classifier"""
    def __init__(self, H, W, in_channels = 1, num_classes = 5, name = "mul_class", backbone = "simple"):
      """Input:
      + H,W
      + in_channels: 1 for greyscale, 3 for RGB
      + num_classes
      + backbone: either 'simple' or 'resnet18'
      """
      super().__init__(H, W, in_channels, name, backbone)
      self.num_classes = num_classes

      # self.feature_extractor defined in parent class
      # self.flattener defined in parent class

      # Number of channels after the Flatten() 
      n_channels_flattened = self.get_flatten_channels(H, W)

      self.classifier = self._build_classifier_head(n_channels_flattened)

      # Need to specify those before calling step()
      # self.loss_calc = None
      # self.optimizer = None
    
    def _build_simple_f_extractor(self):
      """Private function: simple 3 conv module for f extractor"""
      feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels=96, 
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
    
    def _build_classifier_head(self, n_channels):
      """Input: number of channels calculated after flattner & number of multilabel classes"""
      return nn.Sequential(
        nn.Linear(n_channels, 96),
        nn.Dropout(0.3),
        nn.ReLU(),

        nn.Linear(96, 48),
        nn.Dropout(0.3),
        nn.ReLU(),

        nn.Linear(48, self.num_classes),
        nn.Sigmoid()
      )
