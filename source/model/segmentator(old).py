import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from source.model.conv_net import * 
import segmentation_models_pytorch as smp

class SemanticSegmentator(ConvolutionalNetwork): 
    """Input: (B, in_channels, H, W),
    encoder_name: Refer to https://smp.readthedocs.io/en/latest/encoders.html for full list of encoders
    architecture_name: one of ["Unet", "DeepLabV3+"]
    Output: logits (B, num_classes, H, W) WITHOUT softmax activation"""
    
    def __init__(self, H, W, in_channels = 1, num_classes = 5, 
                 architecture_name = "Unet", encoder_name="resnet18", name = "segmentator"):
      super().__init__(H, W, name)
      self.in_channels = in_channels
      self.num_classes = num_classes
      self.encoder_name = encoder_name
      self.architecture_name = architecture_name

      model = self.get_model_factory()

      #Split the model into encoder, decoder and segmentation_head
      self.encoder = model.encoder
      self.decoder  = model.decoder 
      self.segmentation_head = model.segmentation_head 

    def get_model_factory(self):
      """Get the corresponding model to self.architecture_name and self.encoder_name
      Design pattern: Factory"""
      if self.architecture_name == "Unet":
        return smp.Unet(encoder_name = self.encoder_name, in_channels = self.in_channels,
                        classes = self.num_classes)
      elif self.architecture_name == "DeepLabV3+":
        return smp.DeepLabV3Plus(encoder_name = self.encoder_name, in_channels = self.in_channels,
                        classes = self.num_classes)
      
      else:
        raise Exception("Architecture {0:s} has not been implemented. Please select among [Unet, DeepLabV3+]".format(self.architecture_name))
        return None

    def forward(self, x):
      """Extract both logits and features for segmentation"""
      out = {}

      x = self.encoder(x)
      out['features'] = x

      x = self.decoder(*x)
      out['decoder_output'] = x

      x = self.segmentation_head(x)
      out['logits'] = x
            
      return out
