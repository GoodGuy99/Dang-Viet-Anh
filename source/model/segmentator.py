import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from source.model.conv_net import * 
import segmentation_models_pytorch as smp
from keras_unet_collection import models 

class SemanticSegmentator(ConvolutionalNetwork): 
    """Input: (B, in_channels, H, W),
    encoder_name: Refer to https://smp.readthedocs.io/en/latest/encoders.html for full list of encoders
    architecture_name: one of ["Unet", "DeepLabV3+"]
    Output: logits (B, num_classes, H, W) WITHOUT softmax activation"""
    
    def __init__(self, H, W, in_channels = 1, num_classes = 5, 
                 architecture_name = "Unet", encoder_name="resnet18", name = "segmentator",
                 input_size = (256, 1600, 3) ,
                 filter_num_begin = 64 ,
                 n_labels = 5 , 
                 depth = 4 ,
                 stack_num_down = 2 ,
                 stack_num_up = 2 ,
                 patch_size = (2, 2) ,
                 num_heads = [4, 8, 8, 8] ,
                 window_size = [4, 2, 2, 2] ,
                 num_mlp = 512 ,
                 output_activation = 'Softmax' ,
                 shift_window = True 
                 ):
      super().__init__(H, W, name)
      self.in_channels = in_channels
      self.num_classes = num_classes
      self.encoder_name = encoder_name
      self.architecture_name = architecture_name

      self.input_size = input_size
      self.filter_num_begin = filter_num_begin
      self.n_labels = n_labels
      self.depth = depth
      self.stack_num_down = stack_num_down
      self.stack_num_up = stack_num_up
      self.patch_size = patch_size
      self.num_heads = num_heads
      self.window_size = window_size
      self.num_mlp = num_mlp
      self.output_activation = output_activation
      self.shift_window = shift_window

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
      elif self.architecture_name == "SwinUnet":
        return models.swin_unet_2d(input_size = self.input_size, filter_num_begin=self.filter_num_begin, 
                                   n_labels=self.n_labels, depth=self.depth, 
                                   stack_num_down=self.stack_num_down, 
                                   stack_num_up=self.stack_num_up, 
                                   patch_size=self.patch_size, 
                                   num_heads=self.num_heads, window_size=self.window_size, 
                                   num_mlp=self.num_mlp, 
                                   output_activation=self.output_activation, 
                                   shift_window=self.shift_window)
      
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
