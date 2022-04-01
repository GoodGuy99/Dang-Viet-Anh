from PIL import Image
import numpy as np
from torchvision.transforms import *
import torch
from torch.utils import data
import os
from source.data.master_dataset import MasterDataset, MasterDatasetv2

class BinaryDataset(MasterDataset):

    def __init__(self, bin_df, image_folder, H,W, IMG_MEAN = 0, IMG_STD = 1, aug = [], split='train'):
        """BinaryDataset input:
        + bin_df: the dataframe of the binary dataset, which includes ['ImageId','ClassId'] on split 'train'/'val' and ['ImageId'] on split 'test'
        + image_folder: relative or abs. path to the image folder 
        + split: 'train'/'val' or 'test' 
        + IMG_MEAN, IMG_STD: mean and standard deviation for normalization. Get in config filed
        + aug: one of [], ['light'], ['medium'] or ['light', 'medium']
        """
        super().__init__(bin_df, image_folder,H,W, IMG_MEAN, IMG_STD, aug, split)      
        
    def __getitem__(self, index):
        #image logic is stored in parent class
        image, image_name = self.__get_image__(index)

        row = self.df.iloc[index, :] 

        if self.split != 'test':
          label = torch.tensor(row['ClassId'])
          label = torch.unsqueeze(label, dim=0)
          return image, label, image_name
        else:
          return image, image_name

################################
class BinaryDatasetv2(MasterDatasetv2):

    def __init__(self, seg_df, image_folder, H, W, IMG_MEAN = 0, IMG_STD = 1, aug = [], split='train'):
      """BinaryDataset input:
      + bin_df: the dataframe of the binary dataset, which includes ['ImageId','ClassId'] on split 'train'/'val' and ['ImageId'] on split 'test'
      + image_folder: relative or abs. path to the image folder 
      + split: 'train'/'val' or 'test' 
      + IMG_MEAN, IMG_STD: mean and standard deviation for normalization. Get in config filed
      + aug: one of [], ['light'], ['medium'] or ['light', 'medium']
      TODO: Does not allow randcrop in binary's aug
      """
      super().__init__(seg_df, image_folder, H, W,IMG_MEAN, IMG_STD, aug , split)      
        
    def __getitem__(self, index):
      """ Get these items and put them into an unordered dict:
      + image: image tensor. May be augmented in 'train' split
      + label: label tensor. Only for train/val split.
      + image_name
      + original_np_image: to display in evaluation & get rid of inverse transform
      """
      return_dict = {}

      row = self.df.iloc[index, :]
      image_name = row['ImageId'] 
      return_dict["image_name"] = image_name

      image_path = os.path.join(self.image_folder, image_name)
      image = Image.open(image_path).convert('L')    
      image = np.array(image, dtype=np.float)
      return_dict["original_np_image"] = image

      # which augs to apply is defined in parent class. No aug/val,test split => still resize & Normalize
      transformed = self.transform(image = image) # which augs to apply is defined in parent class
      image = transformed['image']
      # print(image)

      if self.split == 'test': #(image, image_name, original_PIL_image)
        pass
      else: # train/val => (image, label, image_name, original_PIL_image)
        label = torch.tensor(row['ClassId'])
        label = torch.unsqueeze(label, dim=0) # so that the channel of prediction is 1
        return_dict["label"] = label
      
      # All three splits: Convert (H,W) => (1,H,W)
      return_dict["image"] = np.expand_dims(image, axis = 0)
      return return_dict