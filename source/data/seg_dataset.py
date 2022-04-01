from PIL import Image
import numpy as np
from torchvision.transforms import *
import torch
from torch.utils import data
import os
from source.data.master_dataset import MasterDataset, MasterDatasetv2

class SegmentationDataset(MasterDataset):
    """Segmentation input:
    + seg_df: the dataframe of the segmentation dataset, which includes 
    ['ImageId','Mask_1',...'Mask_4'] on split 'train'/'val' and ['ImageId'] on split 'test'
    Mask_i is in the form of run length encoding pixel lists; filling top to bottom, then left to right

    + image_folder: relative or abs. path to the image folder 
    + H,W: height, width of image & label
    + split: 'train'/'val' or 'test' 
    + IMG_MEAN, IMG_STD: mean and standard deviation for normalization. Get in config file
    + aug: one of [], ['light'], ['medium'] or ['light', 'medium']
      """

    def __init__(self, seg_df, image_folder, H, W, IMG_MEAN = 0, IMG_STD = 1, aug = [], split='train'):
        super().__init__(seg_df, image_folder, H,W,IMG_MEAN, IMG_STD, aug , split)      
        
    def __getitem__(self, index):
        #image logic is stored in parent class
        image, image_name = self.__get_image__(index)

        if self.split != 'test':
          row_masks = self.df.iloc[index, [2,3,4,5]] 
          # print(row_masks)

          label = self.decode_rle_mask(row_masks)
          label = torch.tensor(label, dtype=torch.int8)
          return image, label, image_name
        else:
          return image, image_name

    def decode_rle_mask(self, row_masks):
        '''Decode the run_length_encoding from encoded pixels id
        Given a row, return mask (H, W) from the dataframe `df`
        Value: from 0 to 5'''
        mask = np.zeros(self.H * self.W, dtype=np.uint8)

        for idx, label in enumerate(row_masks.values):
          defect_class = idx+1 #map from 0..3 -> 1..4
          if label is not np.nan:
              label = label.split(" ")
              positions = map(int, label[0::2])
              length = map(int, label[1::2])
              for pos, le in zip(positions, length):
                  mask[pos:(pos + le)] = defect_class
        # print(mask) 
        mask = mask.reshape(self.H, self.W, order='F')
        # print(np.unique(mask))
        return mask
#############################################
import albumentations as A

class SegmentationDatasetv2(MasterDatasetv2):

  def __init__(self, seg_df, image_folder, H, W, IMG_MEAN = 0, IMG_STD = 1, aug = [], split='train'):
    super().__init__(seg_df, image_folder, H, W, IMG_MEAN, IMG_STD, aug, split)      
  
  def __getitem__(self, index):
    """ Get these items and put them into an unordered dict:
    + image: image tensor. May be augmented in 'train' split
    + label: label tensor. Only for train/val split. Maybe augmented in 'train' split
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

    if self.split == 'test': #(image, image_name, original_PIL_image)
      transformed = self.transform(image = image)
      return_dict["image"] = transformed['image']
    else: # train/val => (image, label, image_name, original_PIL_image)
      row_masks = self.df.iloc[index, [2,3,4,5]] 
      label = self.decode_rle_mask(row_masks) #np array already

      # which augs to apply is defined in parent class. No aug/val split => still resize & Normalize
      transformed = self.transform(image = image, mask = label) 
      return_dict["image"] = transformed['image']
      return_dict["label"] = transformed['mask']
    
    # All three splits: Convert (H,W) => (1,H,W)
    return_dict["image"] = np.expand_dims(return_dict["image"], axis = 0)
    return return_dict

  def decode_rle_mask(self, row_masks):
        '''Decode the run_length_encoding from encoded pixels id
        Given a row, return mask (H, W) from the dataframe `df`
        Value: from 0 to 5'''
        mask = np.zeros(self.H * self.W, dtype=np.uint8)

        for idx, label in enumerate(row_masks.values):
          defect_class = idx+1 #map from 0..3 -> 1..4
          if label is not np.nan:
              label = label.split(" ")
              positions = map(int, label[0::2])
              length = map(int, label[1::2])
              for pos, le in zip(positions, length):
                  mask[pos:(pos + le)] = defect_class
        # print(mask) 
        mask = mask.reshape(self.H, self.W, order='F')
        # print(np.unique(mask))
        return mask
