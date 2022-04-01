import torch
from torchvision.transforms import *
from PIL import Image 

palette = [0, 0, 0, # 0 = no defect - black
           255, 255, 0,     # class 1 - yellow
           73, 255, 0,     # class 2 - green
           62, 0, 95,     # class 3 - purple
           255, 166, 213]   # class 4 - pink

zero_pad = 256 * 3 - len(palette) #paletter needs to have length of 256 * 3
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(segmentation_mask):
  """Input: tensor with shape (H,W), value in range [0..4].
  Output: PIL Image with above palette
  Supported data types: {uint8, int16, uint32, float32}"""
  mask = segmentation_mask.unsqueeze(0).type(torch.uint8)
  mask = transforms.ToPILImage()(mask)
  mask.putpalette(palette)
  return mask

print("colorize_mask defined")
