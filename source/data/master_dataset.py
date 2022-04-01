from PIL import Image
import numpy as np
from torchvision.transforms import *
import torch
from torch.utils import data
import os
import albumentations as A

class MasterDataset(data.Dataset):
    """Parent class of all child dataset

    Inherited functionality list:
    x Normalization
    x Get image
    x Oversampling: handled by data_pipeline package
    - Augmentation
      """

    def __init__(self, df, image_folder, H,W, IMG_MEAN = 0, IMG_STD = 1, aug = [], split='train'):
      """ Input:
      + image_folder: relative or abs. path to the image folder 
      + df: dataframe (bin/mul/seg)
      + split: 'train'/'val' or 'test' 
      + IMG_MEAN, IMG_STD: mean and standard deviation for normalization. Get in config file
      + aug: one of [], ['light'], ['medium'] or ['light', 'medium']
      """ 
      self.df = df
      self.image_folder = image_folder
      self.split = split
      self.aug = aug

      #Only used in SegmentationDataset, but need to input it for common dataset interface
      self.H = H 
      self.W = W 
      
      #Transformation           
      transform_list = [transforms.ToTensor(),
                        transforms.Normalize((IMG_MEAN, ), (IMG_STD, ))
                        ]   
      self.transform = transforms.Compose(transform_list)
        
        
    def __get_image__(self, index):
      """Common function: get an transformed image, given the index"""
      row = self.df.iloc[index, :] #get the row that has index i
      image_name = row['ImageId'] 
      image_path = os.path.join(self.image_folder, image_name)

      image = Image.open(image_path).convert('L') #Convert to grayscale mode
      image = np.array(image, dtype=np.float)
      image = self.transform(image)

      return image, image_name

        
    def __getitem__(self, index):
      pass
    
    def __len__(self):
      return len(self.df)
########################################################
import albumentations as A

class MasterDatasetv2(data.Dataset):
    """Implement additional functionalities versus v1:
    - Augmentation by Albumentions
    - return additional original_np_image: to get rid of inverse transform
    - __get__item() return to a dict {"image": ... , "label": ...,} instead of a batch of objects (as in ProDA)
     TODO: Change client training code after this
      """

    def __init__(self, df, image_folder, H,W, IMG_MEAN = 0, IMG_STD = 1, aug = [], split='train'):
      """ + df: dataframe processed by the pipeline
      + image_folder
      + H, W
      + IMG_MEAN, IMG_STD: for normalization
      + aug: one of [], ['light'], ['medium'] or ['light', 'medium']
      + split: one of 'train', 'val', 'test'
      """
      self.df = df
      self.image_folder = image_folder
      self.split = split

      # Only augment for train split
      if self.split != 'train':
        self.aug = []
      else:
        self.aug = aug

      # Only used in SegmentationDataset, but need to input it for common dataset interface
      self.H = H 
      self.W = W 
      
      #Transformation    
      self.IMG_MEAN = IMG_MEAN
      self.IMG_STD = IMG_STD
      self.transform = self.get_transform()
        
    def get_transform(self):
      """Get the Albumentations transform based on self.aug"""
      transform_list = [A.Resize(self.H, self.W)] # fix size of image to stabilize dimension

      if 'light' in self.aug:
        # Rigid transformations: preserves image (Light)
        light_transform_list = [
          A.RandomCrop(width = int(0.7*self.W), height = int(0.7*self.H), p = 0.5),
          A.HorizontalFlip(p = 0.5),
          A.VerticalFlip(p = 0.5),
          A.Resize(self.H, self.W)
        ]
        transform_list += light_transform_list

      if 'medium' in self.aug:
        # Non-rigid transofrmations: distort image a bit (Medium)                    
        medium_transform_list = [
          A.GridDistortion(p=0.5),
          A.RandomGamma(gamma_limit=(90, 110), p=0.5)
        ]
        transform_list += medium_transform_list
      
      # Normalize image after aug (mask is not normalized by default). 
      # Need to add max_pixel_value to mimic torchvisions.transforms.Normalize
      transform_list += [ A.Normalize(mean = self.IMG_MEAN, std = self.IMG_STD, max_pixel_value = 1) ]

      transform = A.Compose(transform_list)
      return transform
        
    def __getitem__(self, index):
      """Need to be implemented by child class"""
      raise NotImplementedError()
    
    def __len__(self):
      return len(self.df)
